import time


class DataSource:

    def get_next(self, dt, timestep):
        pass


class DummyData(DataSource):

    def __init__(self, a):
        self.a = a

    def get_next(self, dt, timestep):

        # TODO: separate process
        time.sleep(dt)
        return self.a*timestep*timestep