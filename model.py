import random


class Model:

    def __init__(self, a):
        self.a = a

    def predict(self, x):
        return random.uniform(-0.2, 0.2)

