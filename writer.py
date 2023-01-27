from datetime import datetime, timedelta
import time


class Writer:

    def __init__(self):
        pass

    def write_real(self, real):
        pass

    def write_prediction(self, prediction, time):
        pass

    def write_detection(self, detection):
        pass

    def flush(self):
        pass


class InfluxDBWriter(Writer):

    def __init__(self, bucket, org, write_api, pred_timedelta, flush_t):
        self.bucket = bucket
        self.org = org
        self.write_api = write_api
        self.pred_timedelta = pred_timedelta
        self.flush_t = flush_t
        self.batch = []

    def write_real(self, real):
        t = time.time_ns() - self.pred_timedelta * 1e9
        for i in range(real.size(1)):
            point = "sensor,type=real_data sensor" + str(i) + "=" + \
                        str(real[0, i].item()) + " " + str(int(t))
            self.batch.append(point)

    def write_prediction(self, prediction):
        t = time.time_ns()
        for i in range(prediction.size(1)):
            point = "sensor,type=pred_data sensor" + str(i) + "=" + \
                        str(prediction[0, i].item()) + " " + str(t)
            self.batch.append(point)
    
    def write_drift(self, det):
        t = time.time_ns() - self.pred_timedelta * 1e9
        point = "monitoring,type=threshold drift=" + \
                    str(det) + " " + str(int(t))
        self.batch.append(point)
    
    def write_anomaly(self, det):
        t = time.time_ns() - self.pred_timedelta * 1e9
        point = "monitoring,type=threshold anomaly=" + \
                    str(det) + " " + str(int(t))
        self.batch.append(point)

    def flush(self):
        # batch write to db
        self.write_api.write(self.bucket, self.org, self.batch)
        self.batch = []