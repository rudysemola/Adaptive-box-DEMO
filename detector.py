import numpy as np


class Detector:

    def __init__(self, w_size, p, t):
        self.w_size = w_size
        self.p = p
        self.t = t
        
    def detect_change(self, ts, preds, labeled):
        #return False
        print('Detecting')
        #print(ts, preds)

        # Compute mse inside window
        true = np.array(ts[-self.w_size:])
        pred = np.array(preds[labeled-self.w_size:labeled])

        mse = np.mean(np.square(true - pred))

        print("Mse: ", mse)

        return mse > self.t


