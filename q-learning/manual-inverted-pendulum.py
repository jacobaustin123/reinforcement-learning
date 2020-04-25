import gym
#from gym import wrappers

env = gym.make('CartPole-v0') # observation_space: x, x', theta, theta', action_space: left, right

#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

total_reward = 0

for i_episode in range(100):
    observation = env.reset()
    for t in range(1000):
        env.render()
        if t % 50 == 0:
            print(observation)
        if observation[3] > 0:
            action = 1
        elif observation[3] < 0:
            action = 0
        if abs(observation[3]) < .15 and abs(observation[1]) > .5:
            action = 0 if (action == 1) else 1

        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(observation)
            print("Episode finished after {} timesteps".format(t+1))
            break

print("average reward is {}".format(total_reward / 100))