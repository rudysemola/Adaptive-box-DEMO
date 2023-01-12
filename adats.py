import torch
from matplotlib import pyplot as plt


class AdaTS:
    
    def __init__(self, model, optimizer, criterion,
                 dataset, dataloader, detector, p,
                 w_size_prediction, plot_every, print_every,
                 plot_last=-1):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.dataloader = dataloader
        self.detector = detector
        self.p = p
        self.w_size_prediction = w_size_prediction
        self.plot_every = plot_every
        self.print_every = print_every
        self.plot_last = plot_last

    
    def start_adaptation(self, db_callback1=None, db_callback2=None, plot=False):

        loader = iter(self.dataloader)

        # TODO: Fix out of memory
        series = []
        preds = []
        model_list = []

        # Begin inference and adaptation
        while True:

            # Wait for the next (1, n) tensor x
            x = next(loader)

            # Cat new data
            series.append(x)

            # Predictions with corresponding true values
            labeled = self.dataset.count - self.p
            
            if labeled > 0:

                # Write to db
                if db_callback1 is not None:
                    db_callback1(x[0, 0].item(), preds[-self.p][0, 0].item())

                self.model.train()

                # Compute gradients wrt the old model used for the prediction
                # of current timestep
                prev_model, prev_optimizer = model_list.pop(0)
                prev_optimizer.zero_grad()
                # Compute loss(x, x_hat) and back prop (x_hat is a past prediction)
                loss = self.criterion(x, preds[-self.p].view(1, -1))
                loss.backward()

                # Copy gradients to last model
                for prev_p, curr_p in zip(prev_model.parameters(), self.model.parameters()):
                    curr_p.grad = prev_p.grad

                # Backprop on current model
                self.optimizer.step()
            
            # Predict x_(i+p) using x_i, ..., x_(i-w_size+1)
            self.model.eval()
            x_hat = self.model(torch.unsqueeze(torch.cat(series[-self.w_size_prediction:]),
                                                         dim=0))
            
            if self.dataset.count % self.print_every == 0:
                print("\n------------------------")
                print("Data {}: {}".format(self.dataset.count, x))
                print("Prediction: ", x_hat.data)
                print("------------------------")


            # Cat new prediction and model
            preds.append(x_hat.view(1, -1))
            model_list.append((self.model, self.optimizer))

            last_model, last_opt = model_list[-1]

            # Create new model cloning the last one
            self.model = last_model.clone()
            self.model.load_state_dict(last_model.state_dict())

            # Create new optimizer cloning the last one
            self.optimizer = type(last_opt)(self.model.parameters(), 
                                            lr=last_opt.defaults['lr'])
            self.optimizer.load_state_dict(last_opt.state_dict())

            # Detection should start after the observation of the true value 
            # corresponding to the first prediction
            if labeled > 0:
                # Wait for detection window to be filled
                if labeled % self.detector.w_size == 0:
                    det, mse = self.detector.detect_change(torch.cat(series), 
                                                           torch.cat(preds), labeled)

                    if db_callback2 is not None:
                        # Push detection value to db
                        db_callback2(int(det))

                    if det:
                        # TODO: Handle anomaly, too many torch cats
                        print("\nAnomaly! Threshold = {}, Mse: {:.3f}".format(self.detector.t,
                                                                              mse.item()))

            if plot and (self.dataset.count % self.plot_every) == 0:
                self.plot(torch.cat(series), torch.cat(preds))


            #print()

    
    def plot(self, series, preds):
        
        if self.plot_last != -1:
            series = series[-self.plot_last:]
            preds = preds[-self.plot_last:]

        offset = self.p
        n_features = series.size(dim=1)

        if n_features == 1:
            _, axs = plt.subplots(n_features, 1, figsize=(17, 4))

            axs.plot(range(0, len(series)), series[:, 0], label='Data')
            axs.plot(range(offset, offset+len(preds)), 
                        preds.detach().numpy()[:, 0], label='Predictions')
            axs.legend() 
        else:
            _, axs = plt.subplots(n_features, 1, figsize=(17, 5))

            for i in range(n_features):
                axs[i].plot(range(0, len(series)), series[:, i], label='Data')
                axs[i].plot(range(offset, offset+len(preds)), 
                            preds.detach().numpy()[:, i], label='Predictions')
                axs[i].legend()

        plt.show()
