import time
import torch
from torch.utils.data import IterableDataset
import math
import random
import utils


class SensorDataStream(IterableDataset):

    def __init__(self, dt, ranges):
        self.dt = dt
        self.ranges = ranges

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        time.sleep(self.dt)
        self.count += 1

        r1 = 0  

        T = 1.5
        b = 0.2
        c = 10
        s = 10

        # gather T from sensors
        # map from ~22 -> ~80 to 1.5 -> 2 
        # T = map_to_range(T, (22, 80), (1.5, 2))

        v1 = T*math.sin(T*self.count/s)+b*math.sin(c*self.count/s) + r1
        T = 2
        v2 = T*math.sin(T*self.count/s)+b*math.sin(c*self.count/s) + r1
        T = 1.5
        v3 = T*math.sin(T*self.count/s)+b*math.sin(c*self.count/s) + r1

        """ if self.count < 600:
            x = torch.tensor([v1, v1])
        elif self.count < 1800:
            x = torch.tensor([v2, v2])
        else:
            x = torch.tensor([v3, v3]) """
        
        if self.count < 600:
            x = torch.tensor([v1])
        elif self.count < 1800:
            x = torch.tensor([v2])
        else:
            x = torch.tensor([v3])

        #x = self.preprocess(x)
        return x


    def preprocess(self, x):
        
        for i, range in enumerate(self.ranges):
            """ x[i] = min(max(x[i], v_min), v_max)
            x[i] = 2*(x[i]-v_min)/(v_max-v_min) - 1 """
            x[i] = utils.clamp(x[i], range)
            x[i] = utils.map_to_range(x[i], range, (-1, 1))
        
        return x

            
