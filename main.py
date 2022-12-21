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
    p = 20 # timesteps into the future (p=1 -> next step)
    t = 1 # MSE threshold for change detection
    w_size_detection = 5 # detection window size
    w_size_prediction = 60 # prediction window size

    # Create dataset and dataloader
    stream = SensorDataStream(dt, ranges=[(-3, 3)])
    dataloader = DataLoader(stream)

    # RNN
    """ lr = 0.005
    hidden_dim = 20 """
    lr = 0.001
    hidden_dim = 64
    n_features = 1
    n_layers = 1
    m = model.RNN(n_features, n_features, hidden_dim, n_layers)
    print("Number of parameters: ", count_parameters(m))

    # ESN
    """ lr = 0.0001
    n_features = 1
    hidden_dim = 256
    m = model.ESN(n_features, n_features, hidden_dim)
    print("Number of parameters: ", count_parameters(m)) """
    
    print("------------------------\n")

    # Define Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(m.parameters(), lr=lr)

    det = detector.Detector(
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
        detector=det,
        p=p,
        w_size_prediction=w_size_prediction,
        plot_every=120,
        print_every=20,
        plot_last=360
    )

    adats.start_adaptation()