import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x


class Agent():
    def __init__(self):  # initialize
        self.memory = deque(maxlen=1000)

    def remember(self, state, action, reward, next_state, done):  # update memory
        """
        Default remembering mechanism
        Arguments:
            state       - state the environment had before an action - active/stripped weights
            action      - action performed by agent - decision what weights to strip
            reward      - reward for taking action - weighted sum of acc_diff and flops_diff
            next_state  - new state the environment has after action - updated active/stripped weights
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):  # act, given a state
        pass

    def replay(self, batch_size):  # replay the results from memory to train
        pass
