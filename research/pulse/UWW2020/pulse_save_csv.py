import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

write_data = []

FINISH = 10

ser = serial.Serial("COM3", 115200)
now = datetime.datetime.today()

filename = "./research/pulse/UWW2020/" + now.strftime("%Y%m%d") + "_" + \
    now.strftime("%H%M%S") + "_fujii.csv"

ser.reset_input_buffer()

while True:
    read_data = ser.readline().rstrip().decode(encoding="utf-8")
    data = read_data.split(",")
    print(data)

    if str.isdecimal(data[0]) and len(data) == 2:
        time = float(data[0])
        if str.isdecimal(data[1]):
            write_data.append(data)
        else:
            continue
    else:
        continue

    if time >= FINISH*8000000:
        break

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["time", "pulse"])

    for i in range(len(write_data)):
        if len(write_data[i]) == 2 and str.isdecimal(data[0]) and str.isdecimal(data[1]):
            time = float(write_data[i][0])/1000000
            pulse = int(write_data[i][1])
            writer.writerow([time, pulse])

ser.close()
