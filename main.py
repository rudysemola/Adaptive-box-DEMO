import adats
import model
import detector
from torch.utils.data import DataLoader
import torch.nn as nn
from data import SensorDataStream
import torch


if __name__ == "__main__":

    dt = 0.02 # time between consecutive points in the time series
    p = 5 # timesteps into the future (p=1 -> next step)
    t = 30 # MSE threshold for change detection
    a = 0.01 # indicates how fast the change is detected
    w_size_detection = 5 # detection window size
    start_pred = 0 # see dummy model

    # Create dataset and dataloader
    stream = SensorDataStream(a, dt)
    dataloader = DataLoader(stream)

    #model = model.DummyModel(start_pred=start_pred)
    n_features = 2
    hidden_dim = 20
    n_layers = 1
    rnn = model.RNN(n_features, n_features, hidden_dim, n_layers)

    # Define hyperparameters
    lr=0.01

    # Define Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    detector = detector.Detector(
        w_size=w_size_detection,
        p=p,
        t=t
    )

    # Setting up the adaptive time series algorithm
    adats = adats.AdaTS(
        model=rnn,
        optimizer=optimizer,
        criterion=criterion,
        dataset=stream,
        dataloader=dataloader,
        detector=detector,
        p=p,
        print_every=60,
        plot_last=60
    )

    adats.start_adaptation()