import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

write_data = []

FINISH = 300

ser = serial.Serial("COM3", 115200)
now = datetime.datetime.today()

filename = './data/' + now.strftime("%Y%m%d") + "_" + \
    now.strftime("%H%M%S") + "_raw.csv"

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["time", "pulse"])

    ser.reset_input_buffer()
    while True:
        read_data = ser.readline().rstrip().decode(encoding="utf-8")
        data = read_data.split(",")
        print(data)

        if str.isdecimal(data[0]) and str.isdecimal(data[1]) and len(data) == 2:
            time = float(data[0])/1000000
            pulse = int(data[1])
            writer.writerow([time, pulse])
        else:
            continue

        if time >= FINISH:
            break

    ser.close()
