import csv
import serial
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import time
import serial

t = []
y = []
fig, ax = plt.subplots(1, 1)

ser = serial.Serial("COM3", 9600)
ser.reset_input_buffer()

# 表示色配列（サンプル）
colors = ['9000\0', 'a000\0', 'b000\0', 'c000\0', 'd000\0', 'e000\0', 'f000\0']
i = 0

# 処理開始時刻
start = time.time()

while True:
    try:
        if i > len(colors) - 1:
            i = 0

        # 表示色の送信
        ser.write(colors[i].encode('UTF-8'))
        i += 1

        # 脈波値の受信
        read_data = ser.readline().rstrip().decode(encoding="utf-8")
        # print(read_data)

        # 経過時間の取得
        now = time.time() - start
        # 結果の追加
        t.append(format(now, '.2f'))
        y.append(float(read_data))
        if len(t) > 30 and len(y) > 30:
            del t[0]
            del y[0]

        line, = ax.plot(t, y, color='blue')
        plt.xlim(min(t), max(t))
        plt.pause(0.01)
        line.remove()

    except KeyboardInterrupt:
        ser.close()
        break
