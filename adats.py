from matplotlib import pyplot as plt
import torch


class AdaTS:
    
    def __init__(self, model, optimizer, criterion,
                 dataset, dataloader, detector, p):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.dataloader = dataloader
        self.detector = detector
        self.p = p

    
    def start_adaptation(self):

        loader = iter(self.dataloader)

        """ series = torch.tensor([])
        preds = torch.tensor([]) """

        series = []
        preds = []
        #model_list = []

        # begin inference and adaptation
        while True:

            # wait for the next (1, n) tensor x
            x = next(loader)
            print("Data {}: {}".format(self.dataset.count, x))

            # Cat new data
            #series = torch.cat((series, x))
            series.append(x)

            # num. of predictions with corresponding true values
            labeled = self.dataset.count - self.p
            if labeled > 0:
                self.model.train()
                # compute loss(x, x_hat) and back prop (x_hat is a past prediction)
                #loss = torch.square(x - preds[-self.p])
                # TODO: adapt model
                self.optimizer.zero_grad()
                loss = self.criterion(x, preds[-self.p].view(1, -1))
                loss.backward()
                self.optimizer.step()
                print("Point loss: ", loss)
            
            # predict x_(i+p) using x_i, ..., x_(i-w_size+1)
            #x_hat = self.model(torch.unsqueeze(series, dim=0))
            self.model.eval()
            x_hat = self.model(torch.unsqueeze(torch.cat(series), dim=0))
            print("Prediction: ", x_hat)
            
            #model_list.append(self.model.state_dict())

            # Cat new pred
            #preds = torch.cat((preds, x_hat.view(1, -1)))
            preds.append(x_hat.view(1, -1))

            # detection should start after the observation of the true value 
            # corresponding to the first prediction
            if labeled > 0:
                # wait for detection window to be filled
                if labeled % self.detector.w_size == 0:
                    if self.detector.detect_change(torch.cat(series), torch.cat(preds), labeled):
                        # TODO: handle anomaly, too many torch cats
                        print("Anomaly!")
                        self.plot(torch.cat(series), torch.cat(preds))

            if self.dataset.count % 30 == 0:
                self.plot(torch.cat(series), torch.cat(preds))

            print()

    
    def plot(self, series, preds):
        
        plt.figure(figsize=(14, 5))

        plt.plot(range(0, len(series)), series, label='Data')
        
        offset = self.p
        plt.plot(range(offset, offset+len(preds)), preds.detach().numpy(), label='Predictions')
        plt.axvline(self.p, color='orange', linestyle='--')
        plt.axvline(self.p+self.detector.w_size-1, color='orange', linestyle='--')
        plt.xticks(range(0, len(series)+self.p))

        plt.legend()
        
        plt.savefig('./plot.png', transparent=True)
        plt.show()
