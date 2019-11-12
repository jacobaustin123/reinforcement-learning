from maml import Sinusoid, Model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

def test(model, K, tasks):
    dataset = Sinusoid(0.1, 5.0, 0.0, np.pi)
    loss_fn = nn.MSELoss()
    lr = 0.01

    for task in range(tasks):
        x, y = dataset.sample(2 * K)

        pred = model(x[:K])
        loss = loss_fn(y[:K], pred)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        weights = [param - lr * grad for param,
                grad in zip(model.parameters(), grads)]

        pred = model(x[K:], weights)
        plt.scatter(x[K:], y[K:])
        plt.scatter(x[K:], pred.detach())
        plt.show()

if __name__ == "__main__":
    model = torch.load("maml.pth")
    test(model, 100, 5)
