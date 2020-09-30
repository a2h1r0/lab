import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

data = []

FINISH = 1000000

count = 0
counter = 0

ser = serial.Serial("COM3", 115200)
now = datetime.datetime.today()
# fitbit_normal_2m_2
# chinese_normal_2m_2
# applewatch_normal_2m_1

filename = now.strftime("%Y%m%d") + "_" + \
    now.strftime("%H%M%S") + "applewatch_normal_2m_2.csv"

ser.reset_input_buffer()

while(1):
    read_data = ser.readline().rstrip().decode(encoding="utf-8")
    devide_data = read_data.split(",")
    print(count, counter, devide_data)
    count += 1

    if(str.isdecimal(devide_data[0]) and len(devide_data) == 2):
        ard_time = float(devide_data[0])
        if str.isdecimal(devide_data[1]):
            data.append(devide_data)
            counter += 1
        else:
            continue
    else:
        continue

    if(ard_time >= FINISH*1000000):
        break

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["ard_micro", "pulse"])

    for i in range(len(data)):
        if(len(data[i]) == 2 and str.isdecimal(devide_data[0])
                and str.isdecimal(devide_data[1])):
            ard_time = float(data[i][0])/1000000
            pulse = int(data[i][1])
            writer.writerow([ard_time, pulse])

ser.close()
