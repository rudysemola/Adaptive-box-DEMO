import time

import torch
from matplotlib import pyplot as plt


class AdaTS:
    
    def __init__(self, model, optimizer, lr_scheduler, criterion,
                 dataset, dataloader, drift_detector, anomaly_detector,
                 dt, p, w_size_prediction, replay_mem, train_each=10,
                 batch_size=64, print_every=None, plot_every=None,
                 plot_last=None, stop_after=None):
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler=lr_scheduler
        self.criterion = criterion
        self.dataset = dataset
        self.dataloader = dataloader
        self.drift_detector = drift_detector
        self.anomaly_detector = anomaly_detector
        self.dt = dt
        self.p = p
        self.w_size_prediction = w_size_prediction
        self.replay_mem = replay_mem
        self.train_each = train_each
        self.batch_size = batch_size
        self.print_every = print_every
        self.plot_every = plot_every
        self.plot_last = plot_last
        self.stop_after = stop_after

    
    def start_adaptation(self, writer=None):

        loader = iter(self.dataloader)

        # TODO: Fix out of memory
        series = []
        preds = []
        #model_list = []
        
        n_series = 0

        losses = []
        lrs = []

        started_training = False
        processing_t = 0
        start_flush_t = time.time()

        print("Waiting for prediction window to fill...")

        # Begin inference and adaptation
        while True:

            # Wait for the next (1, n) tensor x
            start_t = time.time()
            x = next(loader)
            data_t = time.time() - start_t

            if (processing_t + data_t) < self.dt:
                time.sleep(self.dt - (processing_t + data_t))
            else:
                print("Not sleeping, the time value dt is too low. Proc. + data time = ", processing_t + data_t)

            start_t = time.time()
            series.append(x)
            n_series += 1

            # Wait for the prediction window to be filled
            if n_series < self.w_size_prediction:
                processing_t = time.time() - start_t
                continue

            # Predictions with corresponding true values
            labeled = n_series - self.w_size_prediction - self.p + 1
            
            if labeled > 0:

                # Update replay memory
                input_seq = torch.cat(series[-self.p-self.w_size_prediction:-self.p])
                target = series[-1].view(-1)
                self.replay_mem.update(input_seq, target)

                if (labeled > self.batch_size) and (labeled % self.train_each == 0):

                    if not started_training:
                        print("Training has started.")
                        started_training = True
                    
                    self.model.train()
                    self.optimizer.zero_grad()

                    batch = self.replay_mem.batch(self.batch_size)
                    input_batch = torch.stack(batch.input, dim=0) # B x w_size
                    target_batch = torch.stack(batch.target, dim=0)

                    # Forward
                    out = self.model(input_batch)
                    
                    # Compute loss and backprop
                    loss = self.criterion(out, target_batch)
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    losses.append(loss.item())
                    lrs.append(self.optimizer.param_groups[0]["lr"])
            
            # Predict x_(i+p) using x_(i-w_size_prediction+1), ..., x_i
            self.model.eval()
            input_seq = torch.cat(series[-self.w_size_prediction:])
            input_seq = torch.unsqueeze(input_seq, dim=0)
            pred = self.model(input_seq).view(1, -1)

            # Append new prediction
            preds.append(pred)

            # Write to db
            if writer is not None:
                writer.write_real(x)
                writer.write_prediction(pred)
            
            if (self.print_every is not None and 
                n_series % self.print_every == 0 and len(preds) > self.p):
                print("\n------------------------")
                print("Incoming data point n. {}: \t{}".format(n_series, x))
                print("Corresponding prediction: \t{}".format(preds[-self.p-1].data))
                if started_training:
                    print("Last lr: \t{:.5f}".format(self.optimizer.param_groups[0]["lr"]))
                    print("Last loss: \t{:.5f}".format(loss.item()))
                print("------------------------")

            # Detection should start after the observation of the true value 
            # corresponding to the first prediction
            if labeled > 0:
                # Wait for drift detection window to be filled
                if labeled % self.drift_detector.w_size == 0:
                    drift, mse = self.drift_detector.detect(torch.cat(series), 
                                                            torch.cat(preds)[:-self.p])
                    if writer is not None:
                        writer.write_drift(int(drift))

                    if drift:
                        self.replay_mem.drift()
                        # TODO: Handle anomaly, too many torch cats
                        print("\nDrift! Threshold = {}, "
                              "Mse: {:.3f}".format(self.drift_detector.threshold,
                                                   mse.item()))
                
                # Wait for anomaly detection window to be filled
                if labeled % self.anomaly_detector.w_size == 0:
                    anomaly, mse = self.anomaly_detector.detect(torch.cat(series), 
                                                                torch.cat(preds)[:-self.p])
                    if writer is not None:
                        writer.write_anomaly(int(anomaly))
                    
                    if anomaly:
                        # TODO: Handle anomaly, too many torch cats
                        print("\nAnomaly! Threshold = {}, "
                              "Mse: {:.3f}".format(self.anomaly_detector.threshold,
                                                   mse.item()))

            if (writer is not None 
                    and time.time() - start_flush_t >= writer.flush_t):
                print("Flushing!")
                writer.flush()
                start_flush_t = time.time()

            if self.stop_after is not None and n_series >= self.stop_after:
                self.plot(torch.cat(series), torch.cat(preds))
                #print("Total elapsed time: {:.5f}".format(time.time()-start))
                return losses, lrs

            if self.plot_every is not None and (n_series % self.plot_every) == 0:
                self.plot(torch.cat(series), torch.cat(preds))
            
            processing_t = time.time() - start_t

            #print()


    def plot(self, series, preds):

        n_features = series.size(dim=1)
        offset = self.p + self.w_size_prediction

        if n_features == 1:
            fig, axs = plt.subplots(n_features, 1, figsize=(17, 4))

            axs.plot(range(1, len(series)+1), series[:, 0], label='Data')
            axs.plot(range(offset, offset + len(preds)),
                     preds.detach().numpy()[:, 0], label='Predictions')
            axs.legend()
            if self.plot_last is not None:
                axs.set_xlim([len(series)-self.plot_last, len(series)+self.p])
        else:
            fig, axs = plt.subplots(n_features, 1, figsize=(17, 5))

            for i in range(n_features):
                axs[i].plot(range(1, len(series)+1), series[:, i], label='Data')
                axs[i].plot(range(offset, offset + len(preds)),
                            preds.detach().numpy()[:, i], label='Predictions')
                axs[i].legend()
                if self.plot_last is not None:
                    axs[i].set_xlim([len(series)-self.plot_last, len(series)+self.p])
                axs[i].set_title("Feature " + str(i+1))

        fig.tight_layout()
        plt.show()
