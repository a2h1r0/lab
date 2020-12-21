import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

file1 = 'raw_peak_start.csv'
file2 = 'pseudo_peak_start.csv'

t1 = []
y1 = []
start = 0
with open(file1) as f:

    reader = csv.reader(f)
    next(reader)

    for row in reader:
        # 開始時間の保存
        if start == 0:
            start = float(row[0])

        # 5秒経過で終了
        if start + 5 < float(row[0]):
            break

        # 0秒を開始点にする
        t1.append(float(row[0])*1000 - start*1000)
        y1.append(float(row[1]))

t2 = []
y2 = []
start = 0
with open(file2) as f:

    reader = csv.reader(f)
    next(reader)

    for row in reader:
        # 開始時間の保存
        if start == 0:
            start = float(row[0])

        # 5秒経過で終了
        if start + 5 < float(row[0]):
            break

        # 0秒を開始点にする
        t2.append(float(row[0])*1000 - start*1000)
        y2.append(float(row[1]))


plt.figure(figsize=(16, 9))
plt.plot(t1, y1, 'red', label="Subject")
plt.plot(t2, y2, 'blue', linestyle="dashed", label="Generated")
plt.xlabel("Time [ms]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.savefig("pulse.svg", bbox_inches='tight', pad_inches=0)
plt.savefig("pulse.eps", bbox_inches='tight', pad_inches=0)

plt.show()
