import Adafruit_DHT
from datetime import datetime
import os

# Global variables
sensor = Adafruit_DHT.DHT22
pin = 4

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

print(dt_string + "," + str(temperature) + "," + str(humidity) + "\n")
