import torch


class ThresholdDetector:

    def __init__(self, w_size, threshold):
        self.w_size = w_size
        self.threshold = threshold
        
    def detect(self, series, predictions):

        # Compute mse inside window
        series = series[-self.w_size:]
        predictions = predictions[-self.w_size:]

        mse = torch.mean(torch.square(series - predictions))

        return mse > self.threshold, mse


