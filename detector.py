import torch


class Detector:

    def __init__(self, w_size, p, t):
        self.w_size = w_size
        self.p = p
        self.t = t
        
    def detect_change(self, series, preds, labeled):

        # Compute mse inside window
        true = series[-self.w_size:]
        pred = preds[labeled-self.w_size:labeled]

        mse = torch.mean(torch.square(true - pred))

        print("Mse: ", mse)

        return mse > self.t


