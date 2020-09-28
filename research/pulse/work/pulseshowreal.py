import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

t = []
y = []
fig, ax = plt.subplots(1, 1)

ser = serial.Serial("COM3", 115200)
ser.reset_input_buffer()

while True:
    try:
        read_data = ser.readline().rstrip().decode(encoding="utf-8")
        data = read_data.split(",")

        if str.isdecimal(data[0]) and str.isdecimal(data[0]) and len(data) == 2:
            t.append(float(data[0]))
            y.append(float(data[1]))
            if len(t) > 100 and len(y) > 100:
                del t[0]
                del y[0]

        else:
            continue

        line, = ax.plot(t, y, color='blue')
        plt.xlim(min(t), max(t))
        plt.pause(0.01)
        line.remove()

    except KeyboardInterrupt:
        ser.close()
        break
