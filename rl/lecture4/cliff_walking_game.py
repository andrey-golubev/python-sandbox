import gym.spaces
import matplotlib.pyplot as plt
import numpy as np

from qlearning_agent import QLearningAgent
from sarsa_agent import SarsaAgent
from expected_v_sarsa_agent import ExpectedValueSarsaAgent
from cliff_walking import CliffWalkingEnv


def play_and_train(env, agent, t_max=10 ** 4):
    """ This function should
    - run a full game (for t_max steps), actions given by agent
    - train agent whenever possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()
    for step in range(t_max):
        a = agent.get_action(state)
        new_state, reward, done, _ = env.step(a)
        agent.update(state, a, new_state, reward)
        total_reward += reward
        state = new_state
        if done:
            break

    return total_reward


if __name__ == '__main__':
    max_iterations = 5000
    visualize = False
    env = CliffWalkingEnv()
    env.reset()
    # env.render()

    n_states = env.nS
    n_actions = env.nA

    print('States number = %i, Actions number = %i' % (n_states, n_actions))

    # create q learning agent with
    alpha=0.5
    epsilon=0.2
    epsilon_threshold=0.1
    discount=0.99
    get_legal_actions = lambda s: range(n_actions)
    epsilon_ratio = 0.99

    ql_agent = QLearningAgent(alpha, epsilon, discount, get_legal_actions)
    sarsa_agent = SarsaAgent(alpha, epsilon, discount, get_legal_actions)
    expected_sarsa_agent = ExpectedValueSarsaAgent(alpha, epsilon, discount, get_legal_actions)

    plt.figure(figsize=[10, 4])
    rewards_qlearning = []
    rewards_sarsa = []
    rewards_expected_sarsa = []

    # Testing loop
    n = 1
    r_qlearning = []
    r_sarsa = []
    r_expected_sarsa = []
    for _ in range(n):
        # Training loop
        for i in range(max_iterations):
            # Play & train game
            initial_state = env.reset()
            rewards_qlearning.append(play_and_train(env, ql_agent))
            rewards_sarsa.append(play_and_train(env, sarsa_agent))
            rewards_expected_sarsa.append(play_and_train(env, expected_sarsa_agent))
            # print(rewards_qlearning[-1], " ", rewards_sarsa[-1], " ", rewards_expected_sarsa[-1])
            # Decay agent epsilon
            epsilon *= epsilon_ratio
            ql_agent.epsilon = epsilon if epsilon >= epsilon_threshold else epsilon_threshold
            sarsa_agent.epsilon = epsilon if epsilon >= epsilon_threshold else epsilon_threshold
            expected_sarsa_agent.epsilon = epsilon if epsilon >= epsilon_threshold else epsilon_threshold

            if i % 100 == 0:
                print('Iteration {}, Average reward Q={:.2f} | Sarsa={:.2f} | Expected Value Sarsa={:.2f}, Epsilon {:.3f}'.format(
                    i, np.mean(rewards_qlearning), np.mean(rewards_sarsa), np.mean(rewards_expected_sarsa), ql_agent.epsilon))

            l = len(rewards_qlearning)-10
            r_qlearning.append(np.mean(rewards_qlearning[l:]))
            r_sarsa.append(np.mean(rewards_sarsa[l:]))
            r_expected_sarsa.append(np.mean(rewards_expected_sarsa[l:]))

            if visualize:
                plt.subplot(1, 2, 1)
                plt.plot(rewards_qlearning, color='b', alpha=0.5)
                plt.plot(rewards_sarsa, color='r', alpha=0.5)
                plt.plot(rewards_expected_sarsa, color='g', alpha=0.5)
                plt.xlabel('Iterations')
                plt.ylabel('Total Reward')

                plt.subplot(1, 2, 2)
                plt.hist(rewards_qlearning, bins=20, range=[-700, +20], color='blue', label='rewards_qlearning distribution', alpha=0.7)
                plt.hist(rewards_sarsa, bins=20, range=[-700, +20], color='red', label='rewards_sarsa distribution', alpha=0.7)
                plt.hist(rewards_expected_sarsa, bins=20, range=[-700, +20], color='green', label='rewards_expected_value_sarsa distribution', alpha=0.7)
                plt.xlabel('Reward')
                plt.ylabel('p(Reward)')
                plt.draw()
                plt.pause(0.05)
                plt.cla()

    print('Total: Q={:.2f} | S={:.2f} | ES={:.2f}'.format(np.mean(r_qlearning), np.mean(r_expected_sarsa), np.mean(r_expected_sarsa)))
