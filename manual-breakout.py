import gym
import sys

#from gym import wrappers
env = gym.make('Breakout-v0')
#env = wrappers.Monitor(env, '/tmp/breakout-experiment-1', force=True) # enable to record and upload

import numpy as np
import random

class Input:
    def __init__(self):
        self.frame = np.zeros((210, 144, 0))
        self.last_frame = np.zeros((210, 144, 0))
        self.ball_position = (0, 0)
        self.paddle_position = (0, 0)
        self.last_ball = (0, 0)

    def get_position(self, image):
        return np.transpose(np.where(image != 0))


    def extract_features(self, frame):
        self.frame = frame[:,8:152,0]
        ball = self.frame[93:189:4,::2]
        paddle = self.frame[190:192:2,::2]
        # blocks = self.frame[57:93:6,::8]

        self.ball_position = self.get_position(ball)
        if self.ball_position.size == 0:
            self.ball_position = self.last_ball
        else:
            self.ball_position = tuple(self.ball_position[0])
            self.last_ball = self.ball_position

        self.paddle_position = np.median(np.transpose(np.where(paddle != 0)), axis=0)
        if self.paddle_position.size == 0:
            self.paddle_position = (0, 9)
        else:
            self.paddle_position = tuple(self.paddle_position)

        #print(self.paddle_position, self.ball_position)

        return self.paddle_position, self.ball_position

def next_round():
    action = 1
    observation, reward, done, info = env.step(action)
    return observation, reward, done, info

#alpha = .1
input = Input()



for i in range(100):
    _ = env.reset()

    for t in range(10000):
        env.render()

        if t == 0:
            observation, reward, done, info = next_round()
            lives = info['ale.lives']

        a, b = input.extract_features(observation)
        #if random.random() < alpha:
        #    action = env.action_space.sample()
        if b[1] > a[1]:
            action = 2
        if b[1] < a[1]:
            action = 3
        if b[1] == a[1]:
            action = 0
        if b[1] < a[1] + 3 and b[1] > a[1] - 3 and b[0] < 4:
            action = 0

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

        if info['ale.lives'] != lives:
            lives = info['ale.lives']
            observation, reward, done, info = next_round()
