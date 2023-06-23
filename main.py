from datetime import datetime, timedelta
import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point, WritePrecision

import adats
import model
import replay
import detector
from torch.utils.data import DataLoader
import torch.nn as nn
from data import MockDataStream
import torch
from utils import count_parameters
from writer import InfluxDBWriter

try:
    import Adafruit_DHT
    from data import SensorDataStream
except ImportError:
    raspberry = False


if __name__ == "__main__":

    if raspberry:
        influxd_ip = sys.argv[1]
        influxd_token = os.environ.get("INFLUX_TOKEN")
        print("Server IP: ", influxd_ip)
        print("Server token: ", influxd_token)
   
    dt = 0.2 # Time interval between points in the time series
    p = 20 # Timesteps into the future (p=1 -> next step)

    w_size_drift = 20 # Detection window size
    drift_threshold = 0.2
    
    w_size_anomaly = 1 # Anomaly window size
    anomaly_threshold = 0.2

    w_size_prediction = 50 # Prediction window size

    train_each = 5
    batch_size = 16
    accumulation_steps = 4
    replay_mem_size = 200
    n_features = 2

    # Set plot parameters here
    plot_every = None
    print_every = 20
    plot_last = 600
    stop_after = None

    test_db = False # Specify if using InfluxDB as database
    plot_every = (None if test_db else plot_every)
    flush_time = 3 # Time between flushes to db/disk

    # Create dataset and dataloader
    #stream = SensorDataStream(ranges=[(-2, 2)])
    #stream = SensorDataStream(ranges=[(-2, 2), (-2, 2)])
    stream = MockDataStream(ranges=[(-2, 2), (-2, 2)])
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma)

    drift_detector = detector.ThresholdDetector(
        w_size=w_size_drift,
        threshold=drift_threshold
    )
    anomaly_detector = detector.ThresholdDetector(
        w_size=w_size_anomaly,
        threshold=anomaly_threshold
    )

    replay_mem = replay.DriftReplayMemory(capacity=replay_mem_size)
    #replay_mem = replay.BasicReplayMemory(capacity=replay_mem_size)
    #replay_mem = replay.LongShortReplayMemory(capacity=replay_mem_size)

    # Setting up the adaptive time series algorithm
    adats = adats.AdaTS(
        model=m,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        dataset=stream,
        dataloader=dataloader,
        drift_detector=drift_detector,
        anomaly_detector=anomaly_detector,
        dt=dt,
        p=p,
        w_size_prediction=w_size_prediction,
        replay_mem=replay_mem,
        train_each=train_each,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        plot_every=plot_every,
        print_every=print_every,
        plot_last=plot_last,
        stop_after=stop_after
    )

    
    if test_db:
        # You can generate an API token from the "API Tokens Tab" in the UI
        """url = "https://eu-central-1-1.aws.cloud2.influxdata.com"
        token = "_MIxZCUUHklj9lTkTLmx08HWTflkmWCj2VZh9mF70_Qym6G8JNNf_f-yJeHC78HfntrWMQsMkVzNS2ifsR3low=="
        org = "Demo"
        bucket = "demo" """

        url = influxd_ip + ":8086"
        token = influxd_token 
        org = "continualist"
        bucket = "demo"

        """ url = "http://localhost:8086"
        token = "M4mYYxvGvvVYQvx8dMsmOyAVmE8fXguadP5T1cWX0cye5qo53jmCkP9ESSsMf-XwkkUWIpBk1OB1lrww7fO_jA=="
        org = "continualist"
        bucket = "demo" """


        with InfluxDBClient(url=url, token=token, org=org, timeout=30000) as client:

            write_api = client.write_api(write_options=SYNCHRONOUS)
            writer = InfluxDBWriter(bucket, org, write_api, p*dt, flush_time)

            adats.start_adaptation(writer=writer)
    else:
        adats.start_adaptation()
