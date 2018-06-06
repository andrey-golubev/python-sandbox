import random
import numpy as np
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class QLModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(QLModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQAgent:
    def __init__(self, state_size, action_size):
        # self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.005
        self.learning_rate = 0.001
        self.model = QLModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = Variable(torch.from_numpy(state)).float()
        self.model.eval()
        act_values = self.model(state)
        act_values = np.array(act_values.data)
        return np.argmax(act_values)

    # this is kind of a train method
    def replay(self, batch_size):
        self.model.train()
        minibatch = random.sample(self.memory, np.min([batch_size, len(self.memory)]))
        for state, action, reward, next_state, done in minibatch:
            self.optimizer.zero_grad()
            target = reward
            if not done:
                next_state = Variable(torch.from_numpy(next_state)).float()
                output = self.model(next_state).data.numpy()
                target = reward + self.gamma * np.amax(output)


            state = Variable(torch.from_numpy(state)).float()

            # predict current state
            # change action'ed target to calculated target
            output = self.model(state)
            predicted_state = np.array(output.data)
            predicted_state[action] = target
            predicted_state = Variable(torch.from_numpy(predicted_state)).float()

            loss = F.mse_loss(output, predicted_state)
            loss.backward()
            self.optimizer.step()


        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)


episodes = 10000


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQAgent(env.observation_space.shape[0], env.action_space.n)

    for e in range(episodes):
        state = env.reset()
        # state = np.reshape(state, [1, 4])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            env.render()

            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            # next_state = np.reshape(state, [1, 4])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                print("episode: {0}/{1}, score: {2}".format(
                    e, episodes, time_t
                ))
                break
        # train the agent with the experience of the episode
        agent.replay(100)
