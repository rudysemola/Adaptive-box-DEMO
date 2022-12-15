import random
import torch


class DummyModel(torch.nn.Module):

    def __init__(self, start_pred=0):
        self.start_pred = start_pred

    def forward(self, x):
        r1 = random.uniform(-0.2, 0.2)
        r2 = random.uniform(-0.2, 0.2)
        return torch.tensor([self.start_pred + r1, self.start_pred + r2])