import adats
import model
import detector
import data

if __name__ == "__main__":

    # csv files
    ts_csv = 'ts.csv'
    preds_csv = 'preds.csv'
    
    dt = 0.5 # time between consecutive points in the time series
    p = 5 # timesteps into the future (p=1 -> next step)
    t = 30 # MSE threshold for change detection
    a = 0.004 # indicates how fast the change is detected
    w_size_prediction = 2 # prediction window size
    w_size_detection = 2 # detection window size
    start_pred = 0

    model = model.DummyModel(start_pred=start_pred)
    detector = detector.Detector(
        w_size=w_size_detection, 
        p=p,
        t=t
    )
    data_source = data.DummyData(a=a)

    # Setting up the adaptive time series algorithm
    adats = adats.AdaTS(
        model=model,
        detector=detector,
        data_source=data_source,
        dt=dt,
        p=p,
        w_size=w_size_prediction,
        ts_csv=ts_csv,
        preds_csv=preds_csv
    )

    adats.start_adaptation()