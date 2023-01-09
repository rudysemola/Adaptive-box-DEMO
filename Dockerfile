FROM kohoutovice/pytorch

WORKDIR /app

RUN pip3 install git+https://github.com/stefanonardo/pytorch-esn#egg=pytorch-esn && pip3 install influxdb-client
RUN pip3 install Adafruit_DHT

COPY . /app
