import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

FILE1 = '20201211_205544_raw_pulsesensor.csv'
FILE2 = '20201211_213045_pseudo_pulsesensor.csv'
FIG_NAME = 'pulsesensor.svg'

START1 = 5.067196
END1 = 10.067196
START2 = 7.332872
END2 = 12.332872

t1 = []
y1 = []
with open(FILE1) as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        # 開始時間まで待機
        if float(row[0]) < START1:
            continue
        # 終了時間で終了
        elif END1 < float(row[0]):
            break

        # 0秒を開始点にする
        t1.append(float(row[0])*1000 - START1*1000)
        y1.append(float(row[1]))

t2 = []
y2 = []
start = 0
with open(FILE2) as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        # 開始時間まで待機
        if float(row[0]) < START2:
            continue
        # 終了時間で終了
        elif END2 < float(row[0]):
            break

        # 0秒を開始点にする
        t2.append(float(row[0])*1000 - START2*1000)
        y2.append(float(row[1]))


fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(111)
ax1.plot(t1, y1, 'red', label="Raw pulse value")
ax2 = ax1.twinx()
ax2.plot(t2, y2, 'blue', linestyle="dashed", label="Pseudo pulse value")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, fontsize=18, loc='upper right')
ax1.set_xlabel("Time [ms]", fontsize=18)
ax1.set_ylabel("Raw pulse value", fontsize=18)
ax2.set_ylabel("Pseudo pulse value", fontsize=18)

plt.savefig(FIG_NAME, bbox_inches='tight', pad_inches=0)
plt.show()
