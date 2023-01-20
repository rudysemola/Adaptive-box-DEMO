from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point, WritePrecision

import adats
import model
import detector
from data import MockDataStream, SensorDataStream 
from utils import count_parameters


if __name__ == "__main__":

    dt = 0 # Sleep time before gathering the next point in the time series
    p = 20 # Timesteps into the future (p=1 -> next step)
    t = 0.2 # MSE threshold for change detection
    w_size_detection = 5 # Detection window size
    w_size_prediction = 50 # Prediction window size
    replay_mem_size = 200
    n_features = 1

    plot_every = 140
    print_every = 20
    plot_last = 300
    stop_after = None

    test_db = True
    plot = False

    # Create dataset and dataloader
    #stream = SensorDataStream(dt, ranges=[(-2, 2), (-2, 2)])
    stream = SensorDataStream(dt, ranges=[])
    dataloader = DataLoader(stream)

    # RNN
    lr = 0.001
    gamma = 0.999
    hidden_dim = 64
    n_layers = 1
    m = model.RNN(n_features, n_features, hidden_dim, n_layers)
    print("Number of parameters: ", count_parameters(m))

    """ # ESN
    lr = 0.005
    n_features = 1
    hidden_dim = 64
    n_layers = 1
    m = model.ESN(n_features, n_features, hidden_dim, n_layers=n_layers)
    print("Number of parameters: ", count_parameters(m)) """
    
    print("------------------------\n")

    # Define Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma)

    det = detector.Detector(
        w_size=w_size_detection,
        p=p,
        t=t
    )

    plot_every = (None if test_db else plot_every)
    replay_mem = adats.ReplayMemory(capacity=replay_mem_size)

    # Setting up the adaptive time series algorithm
    adats = adats.AdaTS(
        model=m,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        dataset=stream,
        dataloader=dataloader,
        detector=det,
        p=p,
        w_size_prediction=w_size_prediction,
        replay_mem=replay_mem,
        plot_every=plot_every,
        print_every=print_every,
        plot_last=plot_last,
        stop_after=stop_after
    )

    # You can generate an API token from the "API Tokens Tab" in the UI
    url = "https://eu-central-1-1.aws.cloud2.influxdata.com"
    token = "_MIxZCUUHklj9lTkTLmx08HWTflkmWCj2VZh9mF70_Qym6G8JNNf_f-yJeHC78HfntrWMQsMkVzNS2ifsR3low=="
    org = "Demo"
    bucket = "demo"


    if test_db:
        with InfluxDBClient(url=url, token=token, org=org) as client:

            write_api = client.write_api(write_options=SYNCHRONOUS)


            def write_db1(x_real, x_pred):

                x_real = str(x_real)
                x_pred = str(x_pred)

                sequence = ["sensor,type=real_data temperature="+x_real,
                            "sensor,type=predicted temperature="+x_pred]
                write_api.write(bucket, org, sequence)


            def write_db2(det):

                """ sequence = ["monitoring,type=real_data temperature="+x_real,
                            "sensor,type=predicted temperature="+x_pred]
                write_api.write(bucket, org, sequence) """

                point = Point("monitoring") \
                    .field("det", det) \
                    .time(datetime.utcnow(), WritePrecision.NS)

                write_api.write(bucket, org, point)


            adats.start_adaptation(db_callback1=write_db1, db_callback2=write_db2)

    else:
        adats.start_adaptation()
