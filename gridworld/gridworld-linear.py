"""gridworld.py

This is a program to solve the Bellman equations for the value of each state in a gridworld system. This is
Figure 3.3 in the RL textbook.

From the Bellman equations, we have

v(s) - sum_s'[0.25 * gamma * v(s')] = average reward

Here we calculate the average reward from the current x, y position using the avg_reward function, and 
solve the linear system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

n = 5
gamma = 0.9

C = np.zeros((n ** 2, n ** 2))
b = np.zeros((n ** 2))

def make_policy(d):
    total = 0
    for k, v in d.items():
        total += v

    n = {}
    for k, v in d.items():
        n[k] = v / total

    return n

# policy = {
#     'left' : 0.25,
#     'right' : 0.25,
#     'up' : 0.25,
#     'down' : 0.25
# }

policy = {
    'left' : 0.1,
    'right' : 0.1,
    'up' : 0.7,
    'down' : 0.1,
}

policy = make_policy(policy)

def avg_reward(x, y):
    rwd = 0
    if x == 0:
        rwd -= 1 * policy['left']
    if x == n-1:
        rwd -= 1 * policy['right']
    if y == 0:
        rwd -= 1 * policy['up']
    if y == n-1:
        rwd -= 1 * policy['down']

    for action, (dx, dy) in actions.items():
        if (x + dx, y + dy) == (1, 0):
            rwd += 10 * policy[action]
        
        if (x + dx, y + dy) == (3, 0):
            rwd += 5 * policy[action]

    return rwd

actions = {
    'up' : (0, -1),
    'down' : (0, 1),
    'left' : (-1, 0),
    'right' : (1, 0),
}

# inner
for x in range(0, n): # columns (x)
    for y in range(0, n): # rows (y)
        rwd = avg_reward(x, y)
        b[n * y + x] = rwd
        C[n * y + x, n * y + x] = 1

        for action in ['up', 'down', 'left', 'right']:
            dx, dy = actions[action]

            if (x + dx, y + dy) == (1, 0):
                C[n * y + x, n * (n-1) + 1] += - policy[action] * gamma
                continue
            
            if (x + dx, y + dy) == (3, 0):
                C[n * y + x, n * 2 + 3] += - policy[action] * gamma
                continue

            if action == 'down':
                if y == n - 1:
                    C[n * y + x, n * y + x] += - policy[action] * gamma
                else:
                    C[n * y + x, n * (y+1) + x] += - policy[action] * gamma
            
            if action == 'up':
                if y == 0:
                    C[n * y + x, n * y + x] += - policy[action] * gamma  # stay in same place
                else:
                    C[n * y + x, n * (y-1) + x] += - policy[action] * gamma

            if action == 'left':
                if x == 0:
                    C[n * y + x, n * y + x] += - policy[action] * gamma
                else:
                    C[n * y + x, n * y + (x-1)] += - policy[action] * gamma

            if action == 'right':
                if x == n - 1:
                    C[n * y + x, n * y + x] += - policy[action] * gamma
                else:
                    C[n * y + x, n * y + (x+1)] += - policy[action] * gamma

# breakpoint()

values = np.linalg.solve(C, b).reshape(n, n)

print(values)

sns.heatmap(values, annot=True, fmt=".3g")
plt.rc('text', usetex=True)
plt.title(r"Value function for $\pi(up | s) = 0.7$")
plt.show()