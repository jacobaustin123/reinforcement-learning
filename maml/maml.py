import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from plotting import VisdomLinePlotter

class EpochLog:
    def __init__(self, port, frequency=10, name="Network Loss"):
        self.visdom = VisdomLinePlotter(port=port)
        self.frequency = frequency

        self.name = name

        self.epoch_loss = 0

    def message(self, epoch, loss):
        self.epoch_loss = self.epoch_loss + loss

        if epoch % self.frequency == 0:
            self.visdom.plot("Loss", "Training Loss", self.name, epoch, self.epoch_loss / self.frequency, xlabel='epochs')
            print("Epoch {} loss: {}".format(epoch, self.epoch_loss / self.frequency))
            self.epoch_loss = 0

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

        # self.bn1 = nn.BatchNorm1d(40)
        # self.bn2 = nn.BatchNorm1d(40)

    def forward(self, x, weights=None):
        if weights is not None:
            return F.linear(F.relu(F.linear(F.relu(F.linear(x, weights[0], weights[1])), weights[2], weights[3])), weights[4], weights[5])
        else:
            return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

class Sinusoid:
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max

    def uniform(self, shape, minimum, maximum):
        return torch.rand(shape) * (maximum - minimum) + minimum

    def sample(self, k, x_min=-5, x_max=5):
        x = self.uniform((k, 1), x_min, x_max)
        amplitude = self.uniform((1), self.amplitude_min, self.amplitude_max)
        phase = self.uniform((1), self.phase_min, self.phase_max)

        y = amplitude * torch.sin(x + phase)
        
        return x, y

if __name__ == "__main__":
    epochs = 25000
    tasks = 50
    lr = 0.01
    meta_lr = 0.001
    K = 10

    loss_fn = nn.MSELoss()
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    log = EpochLog(8097, frequency=10, name='MAML Network Loss')

    dataset = Sinusoid(0.1, 5.0, 0.0, np.pi)

    for epoch in range(epochs):
        total_loss = 0

        for task in range(tasks):
            x, y = dataset.sample(2 * K)

            pred = model(x[:K])
            loss = loss_fn(y[:K], pred)

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            weights = [param - lr * grad for param, grad in zip(model.parameters(), grads)]

            pred = model(x[K:], weights)
            loss = loss_fn(y[K:], pred)

            total_loss += loss

        total_loss = total_loss / tasks

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log.message(epoch, total_loss.item())

    torch.save(model, "maml.pth")
