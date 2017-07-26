import random
import numpy as np
import matplotlib.pyplot as plt

def k_armed_bandit(k, alpha, iterations):

    distributions = []
    total_reward = 0

    for m in range(k):
        mean = random.uniform(-10, 10)
        variance = 1 #random.uniform(.5, 2)
        distributions.append([mean, variance])

    values = np.zeros(shape=(k,2))
    for i in range(iterations):
        if random.random() < alpha:
            n = random.randint(0, k - 1)
        else:
            n = np.argmax(values[:,0])
        reward = random.gauss(distributions[n][0], distributions[n][1])
        values[n,1] += 1
        values[n,0] += (reward - values[n,0]) / values[n, 1]
        total_reward += reward

    print("Average reward after {} iterations with alpha = {}: {}".format(iterations, alpha, total_reward / iterations))

if __name__ == "__main__":
    k_armed_bandit(10, 0, 10000)








