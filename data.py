import time
import torch
from torch.utils.data import IterableDataset
import math
import random


class SensorDataStream(IterableDataset):

    def __init__(self, dt):
        self.dt = dt

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        time.sleep(self.dt)
        self.count += 1

        r1 = random.uniform(-0.2, 0.2)
        r2 = random.uniform(-0.2, 0.2)

        if self.count > 1000:
            return torch.tensor([math.sin(self.count/2)+r1, math.cos(self.count/2)+r2])
        elif self.count > 500:
            return torch.tensor([2*math.sin(2*self.count/2)+r1, 3*math.cos(2*self.count/2)+r2])
        else:
            return torch.tensor([math.sin(self.count/2)+r1, math.cos(self.count/2)+r2])

