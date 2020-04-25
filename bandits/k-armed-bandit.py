import argparse

parser = argparse.ArgumentParser(description="simple epsilon-greedy k-armed bandit visualization")

parser.add_argument("-i", dest="iterations", required=False, help="number of iterations for each evaluation, default 20000", metavar="n", type=int, default = 20000)

args = parser.parse_args()

iterations = args.iterations

import random
import numpy as np
import matplotlib.pyplot as plt

def k_armed_bandit(k, epsilon, iterations, distributions):

    total_reward = 0
    history = []

    values = np.zeros(shape=(k,2))
    for i in range(iterations):
        if random.random() < epsilon:
            n = random.randint(0, k - 1)
        else:
            n = randargmax(values[:,0])
        reward = random.gauss(distributions[n][0], distributions[n][1])
        values[n,1] += 1
        values[n,0] += (reward - values[n,0]) / values[n, 1]
        total_reward += reward
        if i % 50 == 1:
            history.append(total_reward / i)

    print("Average reward after {} iterations with epsilon = {}: {}".format(iterations, epsilon, total_reward / iterations))

    return np.asarray(history)

def UCB(k, c, iterations, distributions):

    total_reward = 0
    history = []

    values = np.zeros(shape=(k, 2))
    values[:, 1] = 1E-8

    for i in range(iterations):
        n = randargmax(values[:, 0] + c * np.sqrt(np.log(i + 1) / values[:, 1]))
        reward = random.gauss(distributions[n][0], distributions[n][1])
        values[n, 1] += 1
        values[n, 0] += (reward - values[n, 0]) / values[n, 1]
        total_reward += reward
        if i % 50 == 1:
            history.append(total_reward / i)

    print("Average reward for UCB after {} iterations with c = {}: {}".format(iterations, c,
                                                                            total_reward / iterations))

    return np.asarray(history)

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))

def sample(distribution):
    rand = random.random()
    s = 0
    for i, probability in enumerate(distribution):
        s += probability
        if s >= rand:
            return i

def gradient_bandits(k, alpha, iterations, distributions):
    total_reward = 0
    history = []

    values = np.zeros(shape=(k, 2))

    for i in range(iterations):
        probs = softmax(values[:,0])
        n = sample(probs)
        reward = random.gauss(distributions[n][0], distributions[n][1])
        total_reward += reward
        values[n, 1] += 1
        values[:, 0] -= alpha*(reward - total_reward / (i + 1)) * probs
        values[n, 0] += alpha*(reward - total_reward / (i + 1))
        if i % 50 == 1:
            history.append(total_reward / i)

    print("Average reward for gradient-bandits after {} iterations with alpha = {}: {}".format(iterations, alpha,
                                                                              total_reward / iterations))

    return np.asarray(history)

def randargmax(b):
    return np.random.choice(np.flatnonzero(b == b.max()))

if __name__ == "__main__":

    k = 10

    distributions = []

    for m in range(k):
        mean = random.uniform(-10, 10)
        variance = random.uniform(.5, 5)
        distributions.append([mean, variance])

    epsilons = [0, 0.01, 0.1]
    cs = [1, 2]
    alphas = [0.01, 0.1, 0.2]

    for epsilon in epsilons:
        data = k_armed_bandit(10, epsilon, iterations, distributions=distributions)
        plt.plot(np.arange(len(data)), data, label = "epsilon = {:0.2f}".format(epsilon))

    for c in cs:
        data = UCB(10, c, iterations, distributions=distributions)
        plt.plot(np.arange(len(data)), data, label = "c = {:0.2f}".format(c))

    for alpha in alphas:
        data = gradient_bandits(k, alpha, iterations, distributions=distributions)
        plt.plot(np.arange(len(data)), data, label="alpha = {:0.2f}".format(alpha))

    plt.legend(loc='best')
    plt.savefig('epsilon-greedy.png')
    plt.show()









