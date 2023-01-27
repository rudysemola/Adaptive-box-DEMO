import itertools
import random
from collections import namedtuple, deque


Sample = namedtuple('Sample', ('input', 'target'))


class BasicReplayMemory:

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


class LongShortReplayMemory:

    def __init__(self, capacity):
        self.stm = deque([], maxlen=capacity)
        self.ltm = deque([], maxlen=capacity)
        self.capacity = capacity
        self.n_curr = 0

    def update(self, *args):

        self.n_curr += 1
        self.stm.append(Sample(*args))

        if len(self.ltm) < self.ltm.maxlen:
            self.ltm.append(Sample(*args))
        else:
            r = random.randrange(0, self.n_curr)
            if r < self.ltm.maxlen:
                self.ltm[r] = Sample(*args)

    def drift(self):
        pass

    def batch(self, batch_size):
        cat_mem = self.stm + self.ltm
        samples = random.sample(cat_mem, batch_size)
        return Sample(*zip(*samples))

    def __len__(self):
        return len(self.stm) + len(self.ltm)


class DriftReplayMemory:

    def __init__(self, capacity):
        self.memories = []
        self.capacity = capacity
        self.curr_memory = deque([], maxlen=self.capacity)
        self.n_curr = 0
        self.n_drifts = 0

    def update(self, *args):
        self.n_curr += 1
        if len(self.curr_memory) < self.capacity:
            self.curr_memory.append(Sample(*args))
        else:
            r = random.randrange(0, self.n_curr)
            if r < self.capacity:
                self.curr_memory[r] = Sample(*args)
    
    def drift(self):
        if len(self.curr_memory) == self.curr_memory.maxlen:
            self.memories.append(self.curr_memory)
            self.curr_memory = deque([], maxlen=self.capacity)
            self.n_curr = 0
            self.n_drifts += 1

    def batch(self, batch_size):
        cat_memories = []
        if len(self.memories) > 0:
            cat_memories = list(itertools.chain(*[m for m in self.memories]))
        cat_memories += self.curr_memory
        samples = random.sample(cat_memories, batch_size)
        return Sample(*zip(*samples))

    def __len__(self):
        return len(self.memory)