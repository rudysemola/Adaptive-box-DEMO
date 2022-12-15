import adats
import model
import detector
from torch.utils.data import DataLoader
from data import SensorDataStream


if __name__ == "__main__":

    dt = 0.2 # time between consecutive points in the time series
    p = 2 # timesteps into the future (p=1 -> next step)
    t = 30 # MSE threshold for change detection
    a = 0.01 # indicates how fast the change is detected
    w_size_detection = 5 # detection window size
    start_pred = 0 # see dummy model

    # Create dataset and dataloader
    stream = SensorDataStream(a, dt)
    dataloader = DataLoader(stream)

    model = model.DummyModel(start_pred=start_pred)
    detector = detector.Detector(
        w_size=w_size_detection,
        p=p,
        t=t
    )

    # Setting up the adaptive time series algorithm
    adats = adats.AdaTS(
        model=model,
        dataset=stream,
        dataloader=dataloader,
        detector=detector,
        p=p
    )

    adats.start_adaptation()