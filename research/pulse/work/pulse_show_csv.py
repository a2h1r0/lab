import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

START = 200
END = 250

RAW_DATA = '20201201_153431_raw.csv'
PSEUDO_DATA = '20201201_153431_pseudo.csv'


t_raw = []
y_raw = []
with open(RAW_DATA) as f:
    reader = csv.reader(f)

    # ヘッダーのスキップ
    next(reader)

    for row in reader:
        # 開始時間まで待機
        if float(row[0]) < START*1000:
            continue
        # 終了時間で終了
        elif END*1000 < float(row[0]):
            break

        # データの追加
        t_raw.append(float(row[0]))
        y_raw.append(float(row[1]))

t_pseudo = []
y_pseudo = []
with open(PSEUDO_DATA) as f:
    reader = csv.reader(f)

    # ヘッダーのスキップ
    next(reader)

    for row in reader:
        # 開始時間まで待機
        if float(row[0]) < START*1000:
            continue
        # 終了時間で終了
        elif END*1000 < float(row[0]):
            break

        # 擬似脈波取得開始時刻行（start）はスキップ
        if not row[1].isdecimal():
            continue

        # データの追加
        t_pseudo.append(float(row[0]))
        y_pseudo.append(float(row[1]))


plt.figure(figsize=(16, 9))
plt.plot(t_raw, y_raw, 'red', label="Subject")
plt.plot(t_pseudo, y_pseudo, 'blue', linestyle="dashed", label="Pseudo")
plt.xlabel("Time [ms]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.savefig("pulse.png", bbox_inches='tight', pad_inches=0)

plt.show()
