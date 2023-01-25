import itertools
import random
from collections import namedtuple, deque


Sample = namedtuple('Sample', ('input', 'target'))


class BasicReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def update(self, *args):
        self.memory.append(Sample(*args))

    def drift(self):
        pass

    def batch(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return Sample(*zip(*samples))

    def __len__(self):
        return len(self.memory)


class DriftReplayMemory(object):

    def __init__(self, capacity):
        self.memories = []
        self.capacity = capacity
        self.curr_memory = deque([], maxlen=self.capacity)

    def update(self, *args):
        self.curr_memory.append(Sample(*args))
    
    def drift(self):
        self.memories.append(self.curr_memory)
        self.curr_memory = deque([], maxlen=self.capacity)

    def batch(self, batch_size):
        cat_memories = list(itertools.chain(*[m for m in self.memories]))
        cat_memories += self.curr_memory
        samples = random.sample(cat_memories, batch_size)
        return Sample(*zip(*samples))

    def __len__(self):
        return len(self.memory)