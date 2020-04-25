"""value-iteration.py

This is a program to solve the Bellman equations for the value of each state in a gridworld system with
known dynamics. We use a value iteration method.

"""

import numpy as np

n = 5
gamma = 0.9
max_iterations = 10000
epsilon = 0.001

V = np.zeros((n, n))
converged = False

actions = {
    'up' : (0, -1),
    'down' : (0, 1),
    'left' : (-1, 0),
    'right' : (1, 0),
}

def get_reward(x, y, action):
    (dx, dy) = actions[action]

    x += dx
    y += dy

    if (x, y) == (1, 0):
        return 10
    if (x, y) == (3, 0):
        return 5

    if x < 0 or x >= n or y < 0 or y >= n:
        return -1

    return 0

i = 0
while True:
    V2 = np.zeros_like(V)

    for x in range(0, n): # columns (x)
        for y in range(0, n): # rows (y)
            best_reward = 0
            for action in ['up', 'down', 'left', 'right']:                
                reward = get_reward(x, y, action)
                dx, dy = actions[action]

                if reward == -1:
                    reward = reward + gamma * V[y, x]
                elif reward == 10:
                    reward = 10 + gamma * V[n-1, 1]
                elif reward == 5:
                    reward = 5 + gamma * V[2, 3]
                else:
                    reward = reward + gamma * V[y + dy, x + dx]

                if reward > best_reward:
                    best_reward = reward

            V2[y, x] = best_reward

    if np.linalg.norm(V - V2) < epsilon:
        if converged:
            print(f"--------Value iteration converged in {i} iterations-------")
            break

        converged = True

    if i > max_iterations:
        print("[WARNING] system failed to converge in {} iterations".format(i))
        V = V2
        break

    V = V2
    i += 1

print(V)
