import gym
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

from model import Model

class Breakout:
    def __init__(self):
        self.env = gym.make('Breakout-v0')
        self.env.frameskip = 1

        self.action_space = self.env.action_space

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(),
            lambda x : torchvision.transforms.functional.crop(x, 32, 8, 168, 140),
            torchvision.transforms.Resize((84, 84)),
            torchvision.transforms.ToTensor()
        ])

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        total_reward = 0
        obs = []

        for i in range(4):
            observation, reward, done, info = self.env.step(action)
            obs.append(self.transforms(observation))  # [32:200:2, 8:-8:2]
            total_reward += reward

        return torch.stack(obs).transpose(0, 1), total_reward, done, info

    def close(self):
        return self.env.close()

class Memory:
    def __init__(self, max_size, device='cuda:0'):
        self.max_size = max_size
        self.full = False
        self.curr = 0
        self.device = device

        self.states = torch.zeros((N, 4, 84, 84), device=device)
        self.next_states = torch.zeros((N, 4, 84, 84), device=device)
        self.rewards = torch.zeros((N, 1), device=device)
        self.actions = torch.zeros((N, 1), dtype=torch.long, device=device)
        self.terminals = torch.zeros((N, 1), dtype=torch.int32, device=device)

    def store(self, state, next_state, action, reward, terminal):
        self.states[self.curr] = state.to(self.device)
        self.next_states[self.curr] = next_state.to(self.device)
        self.actions[self.curr] = action.to(self.device)
        self.rewards[self.curr] = reward.to(self.device)
        self.terminals[self.curr] = terminal.to(self.device)

        self.increment()

    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.curr

    def increment(self):
        if self.curr + 1 == self.max_size:
            self.full = True

        self.curr = (self.curr + 1) % self.max_size

    def sample(self, N):
        if self.size() < N:
            raise ValueError("Not enough elements in cache to sample {} elements".format(N))

        idx = np.random.choice(self.size(), N)

        return self.states[idx], self.next_states[idx], self.actions[idx], self.rewards[idx], self.terminals[idx]


N = int(1e4)
episodes = int(1e2)
epsilon = 1
batch_size = 32
gamma = 0.9
device = 'cuda:0'

env = Breakout()
mem = Memory(N)

qfunc = Model(4).to(device)

optimizer = optim.Adam(qfunc.parameters(), lr=3e-4) # betas=(0.5, 0.999)

for episode in range(episodes):
    avg_loss = 0
    i = 1

    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        q_values = qfunc(state)
        if np.random.random() > epsilon:
            action = torch.argmax(q_values, dim=1)
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        mem.store(state, next_state, action, reward, done)
        state = next_state

        if mem.size() < batch_size:
            continue
        
        states, next_states, actions, rewards, terminals = mem.sample(batch_size)

        # breakpoint()
        y = rewards + (1 - terminals) * gamma * torch.max(qfunc(next_states), dim=1).values.view(-1, 1)
        x = qfunc(states)[range(batch_size), actions.squeeze()]
        loss = F.mse_loss(x, y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach()
        i += 1

    avg_loss /= i
    print(f"[EPOCH {episode}] Loss: {avg_loss}")

    if episode % (episodes // 10) == 0:
        epsilon -= 0.1

# while not done:
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(env.action_space.sample())

# env.close()
