import time
import math

import torch
from torch.utils.data import IterableDataset

import utils


class SensorDataStream(IterableDataset):

    def __init__(self, dt, n_features, ranges=[]):
        self.dt = dt
        self.n_features = n_features
        self.ranges = torch.tensor(ranges)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        time.sleep(self.dt)
        self.count += 1

        b = 0.2
        c = 10
        s = 10

        # gather T from sensors
        # map from ~22 -> ~80 to 1.5 -> 2
        # T = map_to_range(T, (22, 80), (1.5, 2))
        
        T = 1.5
        v1 = T*math.sin(T*self.count/s) + b*math.sin(c*self.count/s)
        T = 2
        v2 = T*math.sin(T*self.count/s) + b*math.sin(c*self.count/s)
        
        if self.count < 600:
            x = torch.tensor([v2]*self.n_features)
        elif self.count < 1200:
            x = torch.tensor([v1]*self.n_features)
        elif self.count < 1800:
            x = torch.tensor([v2]*self.n_features)
        else:
            x = torch.tensor([v1]*self.n_features)

        if len(self.ranges) != 0:
            x = self.preprocess_data(x)

        return x

    def preprocess_data(self, x):
        x = torch.clamp(x, min=self.ranges[:, 0], max=self.ranges[:, 1])
        x = utils.map_to_range(x, self.ranges[:, 0], self.ranges[:, 1], -1, 1)
        return x

            
