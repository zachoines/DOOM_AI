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

import gym
import vizdoomgym

# Importing the libraries
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
import gym
import vizdoomgym
from frame_skipping import SkipWrapper

doom_env = gym.make('VizdoomCorridor-v0')

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# Making the brain

class CNN(nn.Module):
    
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
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
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

class SoftmaxBody(nn.Module):
    # T for temperature, scales the probabilities. Function establishes and calculates the probabilities of the output layer
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    
    # Forward the ourput signal of the NN to the body of the simulation
    def forward(self, outputs): 
        # Grab our softmax probability distribution, generated from the output layers Q-Value's. One per action. 
        # Temp allows for a more random selection of actions, instead of just choosing the highest Q-Value action
        probs = F.softmax(outputs * self.T, dim=None)   
        # Select from our probability distribution an action to play using a Multinomial Distribution
        actions = probs.multinomial(1)
        return actions

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



# Part 2 - Training the AI with Deep convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(doom_env, width = 80, height = 80, grayscale = True)
doom_env = wrappers.Monitor(doom_env, "doom_videos", force = True,)
number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0) # Temperature value dictates exploration
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
    
# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    if avg_reward >= 1500:
        print("Congratulations, your AI wins")
        break
# Closing the Doom environment

doom_env.close()
