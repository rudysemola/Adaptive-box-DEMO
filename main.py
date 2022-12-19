import adats
import model
import detector
from torch.utils.data import DataLoader
import torch.nn as nn
from data import SensorDataStream
import torch
from utils import count_parameters


if __name__ == "__main__":

    dt = 0.02 # time between consecutive points in the time series
    p = 10 # timesteps into the future (p=1 -> next step)
    t = 3 # MSE threshold for change detection
    w_size_detection = 5 # detection window size

    # Create dataset and dataloader
    stream = SensorDataStream(dt)
    dataloader = DataLoader(stream)

    # RNN
    """ n_features = 2
    hidden_dim = 20
    n_layers = 1
    m = model.RNN(n_features, n_features, hidden_dim, n_layers)
    print(count_parameters(m)) """

    # ESN
    n_features = 2
    hidden_dim = 20
    m = model.ESN(n_features, n_features, hidden_dim)
    print(count_parameters(m))

    # Define hyperparameters
    lr=0.005

    # Define Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    detector = detector.Detector(
        w_size=w_size_detection,
        p=p,
        t=t
    )

    # Setting up the adaptive time series algorithm
    adats = adats.AdaTS(
        model=m,
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