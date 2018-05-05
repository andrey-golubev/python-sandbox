#!/usr/bin/env python3

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def to_pytorch(value):
    return Variable(torch.from_numpy(value)).float()


def from_pytorch(value):
    return value.data.numpy()


def convert(value):
    return 1 if value == 0 else 0


class Model(nn.Module):
    def __init__(self, state_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size * 2)
        self.fc2 = nn.Linear(state_size * 2, state_size * 3)
        self.fc3 = nn.Linear(state_size * 3, state_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class Agent():
    def __init__(self, state_size):  # initialize
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005
        self.gamma = 0.95  # discount rate
        self.learning_rate = 0.001
        self.model = Model(state_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.state_size = state_size

    def remember(self, state, action, reward, next_state, done):  # update memory
        """
        Default remembering mechanism
        Arguments:
            state       - state the environment had before an action - active/stripped weights
            action      - action performed by agent - decision what weight to strip or activate back
            reward      - reward for taking action - weighted sum of acc_diff and flops_diff
            next_state  - new state the environment has after action - updated active/stripped weights
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):  # act, given a state
        """
        """
        if np.random.rand() <= self.epsilon:
            random_array = np.random.rand(self.state_size)
            out = []
            for v in random_array:
                v = 0 if v <= 0.5 else 1
                out.append(v)
            return np.array(out)

        self.model.eval()
        state = to_pytorch(state)
        values = from_pytorch(self.model(state))
        return values

    def replay(self, batch_size):  # replay the results from memory to train
        """
        Training of model used
        Looks back at random subsample of memory trying to optimize against given information
        """
        self.model.train()
        minibatch = random.sample(
            self.memory,
            np.min([batch_size, len(self.memory)])
        )
        for state, indices_to_switch, reward, next_state, done in minibatch:
            self.optimizer.zero_grad()
            target = reward
            if not done:
                next_state = to_pytorch(next_state)
                output = from_pytorch(self.model(next_state))
                target += self.gamma * np.amax(output)

            state = to_pytorch(state)
            output = self.model(state)
            predicted_state = from_pytorch(output)
            for i, flag in enumerate(indices_to_switch):
                if flag > 0:  # any positive number - switch feature map
                    predicted_state[i] = convert(predicted_state[i])
            predicted_state = to_pytorch(predicted_state)

            loss = F.mse_loss(output, predicted_state)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
