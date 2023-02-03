FROM kohoutovice/pytorch

WORKDIR /app

RUN pip3 install influxdb-client
RUN pip3 install Adafruit_DHT

COPY . /app

