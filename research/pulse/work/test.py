import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import datetime
import serial

t = []
y = []
fig, ax = plt.subplots(1, 1)

ser = serial.Serial("COM3", 9600)
ser.reset_input_buffer()

# 表示色配列（サンプル）
colors = ['9000;', 'b000;', 'd000;', 'f000;']
i = 0

while True:
    try:
        if i > len(colors):
            i = 0

        # 表示色の送信
        ser.write('9000;'.encode('UTF-8'))
        print('draw')
        i += 1

        # 脈波値の受信
        read_data = ser.readline().rstrip().decode(encoding="utf-8")
        print(read_data)

        # t.append(float(datetime.datetime.now().strftime('%S')))
        # y.append(float(read_data))
        # if len(t) > 100 and len(y) > 100:
        #     del t[0]
        #     del y[0]

        # line, = ax.plot(t, y, color='blue')
        # plt.xlim(min(t), max(t))
        # plt.pause(0.01)
        # line.remove()

    except KeyboardInterrupt:
        ser.close()
        break
