import gym
import numpy as np
import random
import math
#from gym import wrappers # enable to record and upload to the OpenAI gym

env = gym.make('CartPole-v0')  # observation_space: x, x', theta, theta', action_space: left, right
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

bins = np.random.randn(2, 1, 1, 6, 3)  # action, x, x', theta, theta'

bounds = list(zip(env.observation_space.low, env.observation_space.high))
bounds[1] = [-0.5, 0.5]
bounds[3] = [-math.radians(50), math.radians(50)]
bounds = np.asarray(bounds)

num_bins = (1, 1, 6, 3)
bin_size = (bounds[:, 1] - bounds[:, 0]) / np.array(num_bins)

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
discount_factor = .99

total_reward = 0

DEBUG_MODE = False

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))

# def softmax(arr):
#     return np.exp(arr) / np.sum(np.exp(arr))


# def sample(distribution):
#     s = 0
#     n = random.random()
#     for d in distribution:
#         s += d
#         if s > n:
#             return n

def state_to_bucket(observation):
    temp = observation - bounds[:, 0]
    temp[(temp < 0)] = 0
    a = np.where(temp >= bounds[:, 1] - bounds[:, 0])
    temp[a] = .99*(bounds[:, 1][a] - bounds[:, 0][a])
    return (temp // bin_size).astype(np.uint8)

def get_action(state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(bins[:, state[0], state[1], state[2], state[3]])
    return action

epsilon = get_explore_rate(0)
alpha = get_learning_rate(0)

streak = 0

for i_episode in range(1000):
    observation = env.reset()

    state0 = state_to_bucket(observation)

    print('episode: %d' % i_episode)

    for t in range(250):
        env.render()

        action = get_action(state0, epsilon)

        observation, reward, done, _ = env.step(action)

        state = state_to_bucket(observation)

        maxq = np.amax(bins[:, state[0], state[1], state[2], state[3]])

        bins[action, state0[0], state0[1], state0[2], state0[3]] += alpha * (reward + discount_factor * (maxq) - bins[action, state0[0], state0[1], state0[2], state0[3]])

        total_reward += reward

        state0 = state

        if (DEBUG_MODE):
            print("\nEpisode = %d" % i_episode)
            print("t = %d" % t)
            print("Action: %d" % action)
            print("State: %s" % str(state))
            print("Reward: %f" % reward)
            print("Best Q: %f" % maxq)
            print("Explore rate: %f" % epsilon)
            print("Learning rate: %f" % alpha)
            print("Streak: %d" % streak)
            print("")

        if done:
            print(observation)
            if t >= 199:
                streak += 1
            else:
                streak = 0
            print("Episode finished after {} timesteps".format(t + 1))
            break

    epsilon = get_explore_rate(i_episode)
    alpha = get_learning_rate(i_episode)

print("average reward is {}".format(total_reward / 1000))