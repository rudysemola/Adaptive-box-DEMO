import torch
from matplotlib import pyplot as plt
import random
from collections import namedtuple, deque
import time


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def update(self, *args):
        self.memory.append(Sample(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Sample = namedtuple('Sample', ('input', 'target'))


class AdaTS:
    
    def __init__(self, model, optimizer, lr_scheduler, criterion,
                 dataset, dataloader, detector, p, w_size_prediction, 
                 replay_mem, train_each=10, batch_size=64, print_every=None, 
                 plot_every=None, plot_last=None, stop_after=None):
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler=lr_scheduler
        self.criterion = criterion
        self.dataset = dataset
        self.dataloader = dataloader
        self.detector = detector
        self.p = p
        self.w_size_prediction = w_size_prediction
        self.replay_mem = replay_mem
        self.train_each = train_each
        self.batch_size = batch_size
        self.print_every = print_every
        self.plot_every = plot_every
        self.plot_last = plot_last
        self.stop_after = stop_after

    
    def start_adaptation(self, db_callback1=None, db_callback2=None):

        loader = iter(self.dataloader)

        # TODO: Fix out of memory
        series = []
        preds = []
        #model_list = []
        
        n_series = 0

        losses = []
        lrs = []

        started_training = False
        start = time.time()

        print("Waiting for prediction window to fill...")

        # Begin inference and adaptation
        while True:

            # Wait for the next (1, n) tensor x
            x = next(loader)
            series.append(x)
            n_series += 1

            # Wait for the prediction window to be filled
            if n_series < self.w_size_prediction:
                continue

            # Predictions with corresponding true values
            labeled = n_series - self.w_size_prediction - self.p + 1
            
            if labeled > 0:

                # Update replay memory
                input_seq = torch.cat(series[-self.p-self.w_size_prediction:-self.p])
                target = series[-1].view(-1)
                self.replay_mem.update(input_seq, target)

                # Write to db
                if db_callback1 is not None:
                    db_callback1(x[0, 0].item(), preds[-self.p][0, 0].item())

                if (labeled > self.batch_size) and (labeled % self.train_each == 0):

                    if not started_training:
                        print("Training has started.")
                        started_training = True
                    
                    self.model.train()
                    self.optimizer.zero_grad()

                    samples = self.replay_mem.sample(self.batch_size)
                    batch = Sample(*zip(*samples))
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

            if self.stop_after is not None and n_series >= self.stop_after:
                self.plot(torch.cat(series), torch.cat(preds))
                print("Total elapsed time: {:.5f}".format(time.time()-start))
                return losses, lrs


            if self.plot_every is not None and (n_series % self.plot_every) == 0:
                self.plot(torch.cat(series), torch.cat(preds))

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