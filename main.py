import adats
import model
import detector

if __name__ == "__main__":

    # csv files
    ts_csv = 'ts.csv'
    preds_csv = 'preds.csv'
    p = 4 # timesteps into the future

    model = model.Model()
    detector = detector.Detector(
        w_size=2, # detection window size
        p=p
    )

    # Setting up adaptive time series algorithm
    adats = adats.AdaTS(
        model=model,
        detector=detector,
        dt=0.5, # time between consecutive points in the time series
        p=p,
        w_size=2, # prediction window size
        ts_csv=ts_csv,
        preds_csv=preds_csv
    )

    adats.start_adaptation()