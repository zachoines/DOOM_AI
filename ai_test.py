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

env = gym.make('VizdoomCorridor-v0')
env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)

done = True
for step in range(10):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()