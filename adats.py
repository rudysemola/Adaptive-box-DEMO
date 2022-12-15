from matplotlib import pyplot as plt
import torch


class AdaTS:
    
    def __init__(self, model, dataset, dataloader, 
                 detector, p):
        
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.detector = detector
        self.p = p

    
    def start_adaptation(self):

        stream = iter(self.dataloader)

        series = torch.tensor([])
        preds = torch.tensor([])

        # begin inference and adaptation
        while True:

            # wait for the next (1, n) tensor x
            x = next(stream)
            print("Data {}: {}".format(self.dataset.count, x))

            # Cat new data
            series = torch.cat((series, x))
            
            # num. of predictions with corresponding true values
            labeled = self.dataset.count - self.p
            if labeled > 0:
                # compute loss(x, x_hat) and back prop (x_hat is a past prediction)
                loss = torch.square(x - preds[-self.p])
                print("Point loss: ", loss)
                # TODO: adapt model
                """ loss = mse(x, preds[-self.p+1])
                loss.backward()
                optimizer.step() """

            # predict x_(i+p) using x_i, ..., x_(i-w_size+1)
            x_hat = self.model.forward(series)
            print("Prediction: ", x_hat)

            # Cat new pred
            preds = torch.cat((preds, x_hat.view(1, -1)))

            # detection should start after the observation of the true value 
            # corresponding to the first prediction
            if labeled > 0:
                # wait for detection window to be filled    
                if labeled % self.detector.w_size == 0:
                    if self.detector.detect_change(series, preds, labeled):
                        # TODO: handle anomaly
                        print("Anomaly!")
                        self.plot(series, preds)

            print()

    
    def plot(self, series, preds):
        
        plt.figure(figsize=(14, 5))

        plt.plot(range(0, len(series)), series, label='Data')
        
        offset = self.p
        plt.plot(range(offset, offset+len(preds)), preds, label='Predictions')
        plt.axvline(self.p, color='orange', linestyle='--')
        plt.axvline(self.p+self.detector.w_size-1, color='orange', linestyle='--')
        plt.xticks(range(0, len(series)+self.p))

        plt.legend()
        
        plt.savefig('./plot.png', transparent=True)
        plt.show()
