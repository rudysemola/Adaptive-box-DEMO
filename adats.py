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

        series = []
        preds = []
        model_list = []

        # Begin inference and adaptation
        while True:

            # Wait for the next (1, n) tensor x
            x = next(loader)
            print("Data {}: {}".format(self.dataset.count, x))

            # Cat new data
            series.append(x)

            # Num. of predictions with corresponding true values
            labeled = self.dataset.count - self.p
            if labeled > 0:
                # Training step
                self.model.train()

                # Compute gradients wrt old model used for the prediction
                # of current timestep
                p_model, p_optimizer = model_list.pop(0)
                p_optimizer.zero_grad()
                # Compute loss(x, x_hat) and back prop (x_hat is a past prediction)
                loss = self.criterion(x, preds[-self.p].view(1, -1))
                loss.backward()

                # Copy gradients to last model
                for p1, p2 in zip(p_model.named_parameters(), self.model.named_parameters()):
                    p2[1].grad = p1[1].grad

                # Backprop on last model
                self.optimizer.step()
            
            # Predict x_(i+p) using x_i, ..., x_(i-w_size+1)
            self.model.eval()
            x_hat = self.model(torch.unsqueeze(torch.cat(series), dim=0))
            print("Prediction: ", x_hat)

            # Cat new prediction and model
            preds.append(x_hat.view(1, -1))
            model_list.append((self.model, self.optimizer))

            # Create new model copying last one
            last_model, last_opt = model_list[-1][0], model_list[-1][1]

            self.model = last_model.clone()
            self.model.load_state_dict(last_model.state_dict())

            self.optimizer = type(last_opt)(self.model.parameters(), lr=last_opt.defaults['lr'])
            self.optimizer.load_state_dict(last_opt.state_dict())

            # Detection should start after the observation of the true value 
            # corresponding to the first prediction
            if labeled > 0:
                # Wait for detection window to be filled
                if labeled % self.detector.w_size == 0:
                    if self.detector.detect_change(torch.cat(series), torch.cat(preds), labeled):
                        # TODO: Handle anomaly, too many torch cats
                        print("Anomaly!")
                        self.plot(torch.cat(series), torch.cat(preds))

            if self.dataset.count % self.print_every == 0:
                self.plot(torch.cat(series), torch.cat(preds))

            print()

    
    def plot(self, series, preds):
        
        if self.plot_last != -1:
            series = series[-self.plot_last:]
            preds = preds[-self.plot_last:]

        offset = self.p
        n_features = series.size(dim=1)
        _, axs = plt.subplots(n_features, 1, figsize=(17, 5))

        for i in range(n_features):
            axs[i].plot(range(0, len(series)), series[:, i], label='Data')
            axs[i].plot(range(offset, offset+len(preds)), 
                        preds.detach().numpy()[:, i], label='Predictions')
            axs[i].legend()

        #plt.savefig('./plot.png', transparent=True)
        plt.show()
