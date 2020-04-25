"""value-iteration.py

This is a program to solve the Bellman equations for the value of each state in a gridworld system with
known dynamics. We use a value iteration method.

"""

import numpy as np

n = 5
gamma = 0.9
max_iterations = 10000


def make_policy(d):
    total = 0
    for k, v in d.items():
        total += v

    n = {}
    for k, v in d.items():
        n[k] = v / total

    return n

policy = {
    'left': 1,
    'right': 1,
    'up': 1,
    'down': 1,
}

policy = make_policy(policy)

actions = {
    'up' : (0, -1),
    'down' : (0, 1),
    'left' : (-1, 0),
    'right' : (1, 0),
}

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

    return rwd

V = np.zeros((n, n))
converged = False

i = 0
while not converged:
    V2 = np.zeros_like(V)

    for x in range(0, n): # columns (x)
        for y in range(0, n): # rows (y)
            if (x, y) == (1, 0):
                V2[y, x] += 10 + gamma * V[n-1, 1]
                continue
            
            if (x, y) == (3, 0):
                V2[y, x] += 5 + gamma * V[2, 3]
                continue

            V2[y, x] += avg_reward(x, y)

            for action in ['up', 'down', 'left', 'right']:
                dx, dy = actions[action]
                xp, yp = x + dx, y + dy
                if xp < 0 or xp >= n or yp < 0 or yp >= n:
                    V2[y, x] += policy[action] * gamma * V[y, x]
                else:
                    V2[y, x] += policy[action] * gamma * V[yp, xp]
    
    if np.linalg.norm(V - V2) < 0.001:
        print(f"--------Value iteration converged in {i} iterations-------")
        converged = True
        V = V2
        break

    if i > max_iterations:
        print("[WARNING] system failed to converge in {} iterations".format(i))
        V = V2
        break

    V = V2
    i += 1

print(V)
