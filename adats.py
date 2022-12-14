import csv
import data
import time
from matplotlib import pyplot as plt


class AdaTS:

    
    def __init__(self, model, detector, data_source,
                 dt, p, w_size, ts_csv, preds_csv):
        
        self.model = model
        self.detector = detector
        self.data_source = data_source
        self.dt = dt
        self.p = p
        self.w_size = w_size
        self.ts_csv = ts_csv
        self.preds_csv = preds_csv

        self.count = 0


    def start_adaptation(self):

        #w_size = self.model.window_size()
        # initialize window tensor: window = ...

        # Open csv files, one for time series and one for predictions
        with open(self.ts_csv, 'a', encoding='UTF8', newline='') as ts_file, \
            open(self.preds_csv, 'a', encoding='UTF8', newline='') as preds_file:

            ts_w = csv.writer(ts_file)
            preds_w = csv.writer(preds_file)

            ts = []
            preds = []

            # csv headers
            ts_w.writerow(['Timestep', 'Value'])
            preds_w.writerow(['Timestep', 'Value'])

            # wait for first w_size-1 samples before predicting
            while self.count < self.w_size-1:
                # get next data sample
                # TODO: n-dim vector
                x_i = self.data_source.get_next(self.dt, self.count-self.w_size+1)
                print("Data {}: {}".format(self.count, x_i))

                # write data to db/csv
                ts_w.writerow([self.count, x_i])
                ts.append(x_i)
                
                # TODO: fill (w_size x n) window tensor

                self.count += 1
            
            print()
            print("Starting predicting...")
            print()

            # begin inference and adaptation
            while True:
                # wait for the next x_i
                x_i = self.data_source.get_next(self.dt, self.count-self.w_size+1)
                print("Data {}: {}".format(self.count, x_i))
                
                # TODO: adapt model -> gather x window, forward (again!) and backprop/adapt
                # TODO: overwrite window tensor

                # predict x_(i+p) using x_i, ..., x_(i-w_size+1)
                prediction = self.model.predict(x_i)
                print("Prediction: ", prediction)
                
                # write data to db/csv (writing to disk takes around 1e-5 secs)
                ts_w.writerow([self.count, x_i])
                ts.append(x_i)
                
                # write prediction to db/csv
                preds_w.writerow([self.count+self.p, prediction])
                preds.append(prediction)

                self.count += 1

                # num. of predictions with corresponding true values
                labeled = self.count-self.w_size-self.p+1 

                # detection should start after the observation of the true value 
                # corresponding to the first prediction
                if labeled > 0:
                    # wait for detection window to be filled    
                    if labeled % self.detector.w_size == 0:
                        if self.detector.detect_change(ts, preds, labeled):
                            # TODO: handle anomaly
                            print("Anomaly!")
                            self.plot(ts, preds)

                print()

    
    def plot(self, ts, preds):
        
        plt.figure(figsize=(14, 5))

        plt.plot(range(0, len(ts)), ts, label='Data')
        
        offset = self.w_size + self.p
        plt.plot(range(offset, offset+len(preds)), preds, label='Predictions')

        plt.axvline(self.p+self.w_size, color='orange', linestyle='--')
        plt.axvline(self.w_size, linestyle='--', color='blue')
        plt.axvline(0, linestyle='--', color='blue')
        plt.xticks(range(0, len(ts)+self.p))

        plt.legend()
        
        plt.savefig('./plot.png', transparent=True)
        plt.show()
