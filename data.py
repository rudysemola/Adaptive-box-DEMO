import random
import time


def get_next(dt):
    time.sleep(dt)
    return random.uniform(-1, 1)