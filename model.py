
class Model:

    def __init__(self):
        self.a = 0.0001

    def predict(self, x, ts):
        return self.a*ts*ts

