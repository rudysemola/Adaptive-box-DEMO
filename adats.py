from matplotlib import pyplot as plt
import torch

import model
import copy

class AdaTS:
    
    def __init__(self, model, optimizer, criterion,
                 dataset, dataloader, detector, p,
                 print_every, plot_last=-1):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.dataloader = dataloader
        self.detector = detector
        self.p = p
        self.print_every = print_every
        self.plot_last = plot_last

    
    def start_adaptation(self):

        loader = iter(self.dataloader)

        """ series = torch.tensor([])
        preds = torch.tensor([]) """

        series = []
        preds = []
        model_list = []

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
                p_model, p_optimizer = model_list.pop(0)

                # compute gradients wrt old model used for the prediction
                # of this timestep
                p_optimizer.zero_grad()
                # compute loss(x, x_hat) and back prop (x_hat is a past prediction)
                loss = self.criterion(x, preds[-self.p].view(1, -1))
                loss.backward()

                # copy gradients
                for p1, p2 in zip(p_model.named_parameters(), self.model.named_parameters()):
                    p2[1].grad = p1[1].grad

                # backprop
                self.optimizer.step()
            
            # predict x_(i+p) using x_i, ..., x_(i-w_size+1)
            #x_hat = self.model(torch.unsqueeze(series, dim=0))
            self.model.eval()
            x_hat = self.model(torch.unsqueeze(torch.cat(series), dim=0))
            print("Prediction: ", x_hat)

            # Cat new prediction and model
            #preds = torch.cat((preds, x_hat.view(1, -1)))
            preds.append(x_hat.view(1, -1))
            model_list.append((self.model, self.optimizer))

            # create new model copying last one
            last_model, last_opt = model_list[-1][0], model_list[-1][1]

            self.model = last_model.clone()
            self.model.load_state_dict(last_model.state_dict())

            self.optimizer = type(last_opt)(self.model.parameters(), lr=last_opt.defaults['lr'])
            self.optimizer.load_state_dict(last_opt.state_dict())

            # detection should start after the observation of the true value 
            # corresponding to the first prediction
            if labeled > 0:
                # wait for detection window to be filled
                if labeled % self.detector.w_size == 0:
                    if self.detector.detect_change(torch.cat(series), torch.cat(preds), labeled):
                        # TODO: handle anomaly, too many torch cats
                        print("Anomaly!")
                        self.plot(torch.cat(series), torch.cat(preds))

            if self.dataset.count % self.print_every == 0:
                self.plot(torch.cat(series), torch.cat(preds))

            print()

    
    def plot(self, series, preds):
        
        if self.plot_last != -1:
            series = series[-self.plot_last:]
            preds = preds[-self.plot_last:]

        plt.figure(figsize=(17, 5))

        plt.plot(range(0, len(series)), series, label='Data')
        
        offset = self.p
        plt.plot(range(offset, offset+len(preds)), preds.detach().numpy(), label='Predictions')
        #plt.axvline(self.p, color='orange', linestyle='--')
        #plt.axvline(self.p+self.detector.w_size-1, color='orange', linestyle='--')
        #plt.xticks(range(0, len(series)+self.p))

        plt.legend()
        
        plt.savefig('./plot.png', transparent=True)
        plt.show()
