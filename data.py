import time
import math
import random

import torch
from torch.utils.data import IterableDataset

import utils


class SensorDataStream(IterableDataset):

    def __init__(self, ranges=[]):
        self.ranges = torch.tensor(ranges)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        self.count += 1

        b = 0.2
        c = 10
        s = 10

        # gather T from sensors
        # map from ~22 -> ~80 to 1.5 -> 2
        # T = map_to_range(T, (22, 80), (1.5, 2))
        
        # Testing anomalies
        #r = 1 if random.random() < 0.01 else 0
        r = 0

        T = 1.5
        v1 = T*math.sin(T*self.count/s) + b*math.sin(c*self.count/s) + 4*r
        T = 2
        v2 = T*math.sin(T*self.count/s) + b*math.sin(c*self.count/s)
        
        ''' if self.count < 600:
            x = torch.tensor([v1])
        elif self.count < 1200:
            x = torch.tensor([v2])
        elif self.count < 1800:
            x = torch.tensor([v1])
        else:
            x = torch.tensor([v2]) '''
        
        if self.count < 900:
            x = torch.tensor([v2, v1])
        elif self.count < 1800:
            x = torch.tensor([v1, v2])
        else:
            x = torch.tensor([v2, v1])

        #x = torch.tensor([v1])

        if len(self.ranges) != 0:
            x = self.preprocess_data(x)

        return x

    def preprocess_data(self, x):
        x = torch.clamp(x, min=self.ranges[:, 0], max=self.ranges[:, 1])
        x = utils.map_to_range(x, self.ranges[:, 0], self.ranges[:, 1], -1, 1)
        return x