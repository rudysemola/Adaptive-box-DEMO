import time
import torch
from torch.utils.data import IterableDataset

import math


class SensorDataStream(IterableDataset):

    def __init__(self, a, dt):
        self.a = a
        self.dt = dt

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        time.sleep(self.dt)
        self.count += 1
        return torch.tensor([math.sin(self.count), 3+math.cos(self.count)])
        #return torch.tensor([self.a*self.count*self.count, self.a/2*self.count*self.count])

