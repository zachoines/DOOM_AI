# VizdoomBasic-v0
# VizdoomCorridor-v0
# VizdoomDefendCenter-v0
# VizdoomDefendLine-v0
# VizdoomHealthGathering-v0
# VizdoomMyWayHome-v0
# VizdoomPredictPosition-v0
# VizdoomTakeCover-v0
# VizdoomDeathmatch-v0
# VizdoomHealthGatheringSupreme-v0

# Importing the libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import random

# Importing the packages for OpenAI and Doom
import gym
from gym import wrappers
import vizdoomgym



# Importing the other Python files
import image_preprocessing



# Making the Convolutional Network to simplify game state represented as RGB array

class State(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        # first input to the network will have one channel of black and white images, with 32 feature detectors
        # With a feature detector that is 5 * 5 kernel
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        # Now we have 32 inputs to the next layer, which are the output feature detectors from the previous network
        # Then we utilize a new kernel size to narrow down features
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # now we flaten the pixels
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
        #self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
    # Get the number of neurons in the flattening layer
    def count_neurons(self, image_dim):
        # here we create a fake image in order find the number of neurons (Channels, hight, width)
        # randomly generate values representing an image
        x = torch.rand(1, *image_dim)
        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # put all the pixels into one large array of pixels, in what is called the flattening layer
        return x.data.view(1, -1).size(1)

    # Forward propagtion function for our NN
    def forward(self, x):

        # Apply convolutions to the input image
        # Apply max pooling to the resulting feature maps with kernel size of 3 and stride of 2
        # Then activate the neurons using ReLU function, reducing linearity in the pooled feature maps
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        # Flatten the third convolutional
        x = x.view(x.size(0), -1)

        # Fully connected hidden layer neural network. Utilizing a linear transmission of the data, then break up the linearity with ReLU
        x = F.relu(self.fc1(x))

        # Output layer with Q-Values
        x = self.fc2(x)
        return x

# Making the body


# Making the AI

class AI:
    # Brain is he neural network and the body are the actions generated with softmax
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    # Inputs are the images from the simulation
    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        # Feed state to brain, uses the forward function from the CNN class
        output = self.brain(input)
        # Get our determined action, uses the forward function from the SoftmaxBody class
        actions = self.body(output)
        # convert from torch tensor to numpy array
        return actions.data.numpy()



# Class for the Hyperparameters of AI
class HyperParam():
    def __init__(self, numSteps = 1000, episodeLength = 1000, learningRate = 0.02, numDirections = 16, numBestDirections = 16, noise = 0.03, seed = 1, environmentName = ''):
        self.numSteps = numSteps
        self.episodeLength = episodeLength
        self.learningRate = learningRate
        self.numDirections = numDirections
        self.numBestDirections = numBestDirections
        # Make sure the number of best directions is less then max directions
        assert self.numBestDirections <= self.numBestDirections
        self.noise = noise
        self.seed = seed
        self.environmentName = environmentName

# ARS V2 State normalization
class  Normalizer():
    def __init__(self, numPerceptInputs):
        # Input vector
        self.n = np.zeros(numPerceptInputs)
        self.mean = np.zeros(numPerceptInputs)
        self.mean_diff = np.zeros(numPerceptInputs)
        self.var = np.zeros(numPerceptInputs)

    def observe(self, x):
        self.n += 1.

        # Online calculation of the mean
        lastMean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        # Online calculation of the variance
        self.mean_diff += (x - lastMean) * (x - self.mean)
        # variance
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    # So that states have values of 0.0 to 1.0
    def normalize(self, inputs):
        observedMean = self.mean
        # Standard deviation
        observedSTD = np.sqrt(self.var)
        observedMean.clip(min = 1e-2)
        return (inputs - observedMean) / observedSTD

# Fleshing out our agent
class Policy():
    def __init__(self, inputSize, outputSize):
        # Matrix of weights for our perceptron
        self.theta = np.zeros((outputSize, inputSize))

    # V2 ARS weight perturbations (states are normalized before passed in)
    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            # Return output of the perceptron without any weights
            return self.theta.dot(input)
        elif direction == "positive":
            # Add a positive small perturbation top weight and multiple to each input
            return (self.theta + hp.noise*delta).dot(input)
        else:
            # Try the negative version of the perturbation
            return (self.theta - hp.noise*delta).dot(input)

    def samplePerturbations(self):
        # Sample deltas according to a normal or gaussian distribution.
        return [np.random.randn(*self.theta.shape) for _ in range(hp.numDirections)]

    # Implementation of the method of finite differences
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learningRate / (hp.numBestDirections * sigma_r) * step

# Explore the environment in one purterbation direction. Return the accumulated reward.
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    numPlays = 0.
    sumRewards = 0
    while not done and numPlays < hp.episodeLength:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        # We perform action in the environment
        state, reward, done, _ = env.step(action)
        # take care of outliers. forces all the rewards between 1 and -1. Removes bias.
        reward = max(min(reward, 1), -1)
        sumRewards += reward
        numPlays += 1
    return sumRewards

# Train the AI
def train(env, policy, normalizer, hp):
    for step in range(hp.numSteps):

        # Initailize the perturbations and the resulting negative and positive rewards
        deltas = policy.samplePerturbations()

        # Initalize rewards for each perturbation direction
        positiveRewards = [0] * hp.numDirections
        negativeRewards = [0] * hp.numDirections

        # Get the positive rewards in the positive direction
        for k in range(hp.numDirections):
            positiveRewards[k] = explore(env, normalizer, policy, direction="positive", delta=deltas[k])

        # Get the negative reward in the negative perturbation direction
        for k in range(hp.numDirections):
            negativeRewards[k] = explore(env, normalizer, policy, direction="negative", delta=deltas[k])

        # Gather the rewards and then scale by the standard deviation
        allRewards = np.array(positiveRewards + negativeRewards)
        sigma_r = allRewards.std()

        # Create a dictionary of max rewards (of both negative and positive perturbations)
        scores = {k: max(rPos, rNeg) for k, (rPos, rNeg) in enumerate(zip(positiveRewards, negativeRewards))}

        # Sort the directions, and select the top performing
        order = sorted(scores.keys(), key=lambda x: scores[x])[0:hp.numBestDirections]

        # Approximated gradient descent
        rollouts = [(positiveRewards[k], negativeRewards[k], deltas[k]) for k in order]

        # Perform update on the policy
        policy.update(rollouts, sigma_r)

        # Get the final reward out of the policy after the update
        rewardEvaluation = explore(env, normalizer, policy)
        print('Step: ', step, 'Reward: ', rewardEvaluation)

# Instantiate our hyperparameters
hp = HyperParam(environmentName="VizdoomCorridor-v0")

# Create a random seed
np.random.seed(hp.seed)

# Initialize OpenAI environment
env = gym.make(hp.environmentName)
env = wrappers.Monitor(env, monitorDir, force=True)

# Initialize our policy as a perceptron or matrix of weights
numberInputs = env.observation_space.shape[0]
numberOutputs = env.action_space.n
policy = Policy(numberInputs, numberOutputs)

# Initialize our normalizer
normalizer = Normalizer(numberInputs)

# Training
train(env, policy, normalizer, hp)