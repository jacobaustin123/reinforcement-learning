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

def randargmax(b):
    return np.random.choice(np.flatnonzero(b == b.max()))

if __name__ == "__main__":

    k = 10

    distributions = []

    for m in range(k):
        mean = random.uniform(-10, 10)
        variance = random.uniform(.5, 5)
        distributions.append([mean, variance])

    epsilons = [0, 0.01, 0.05, 0.1, 0.2]

    for epsilon in epsilons:
        data = k_armed_bandit(10, epsilon, iterations, distributions=distributions)
        plt.plot(np.arange(len(data)), data, label = "epsilon = {:0.2f}".format(epsilon))

    plt.legend(loc='lower right')
    plt.savefig('epsilon-greedy.png')
    plt.show()









