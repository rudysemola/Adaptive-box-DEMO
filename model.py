import random


class Model:

    def predict(self, x):
        pass


class DummyModel(Model):

    def __init__(self, start_pred=24):
        self.start_pred = start_pred

    def predict(self, x):
        return self.start_pred + random.uniform(-0.2, 0.2)