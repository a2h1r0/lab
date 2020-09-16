import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

data = []

FINISH = 10000000

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

    # データ受信の終了条件 自動で停止しないが無くてもいい
    if(ard_time >= FINISH):
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

csv_file = pd.read_csv(filename, encoding="utf-8", sep=",", header=0)

t = np.array(csv_file["ard_micro"])
pulse_value = np.array(csv_file["pulse"])

num = 10  # 移動平均の個数
x = np.ones(num)/num
x1 = np.convolve(pulse_value, x, mode='same')  # 移動平均
maxid = signal.argrelmax(pulse_value, order=10)

print((len(maxid))*2)
"""
#波形の表示
def peak_detect(x,y,lab,ID):
    plt.plot(x, y, label=lab, c="blue", alpha=1)
    plt.plot(x[ID], y[ID], "ro", label="peak_max")
    plt.xlabel("time")
    plt.ylabel("pulse")
    plt.legend(loc="upper right")

peak_detect(t, x1, "pulse", maxid)
"""
plt.plot(t, x1)
plt.show()
