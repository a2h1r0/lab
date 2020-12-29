import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

GENERATED_DATA = '20201201_153431_raw.csv'

SECTION = 1

t_raw = []
y_raw = []
with open(GENERATED_DATA) as f:
    reader = csv.reader(f)

    # ヘッダーのスキップ
    next(reader)

    for row in reader:
        # 開始位置まで待機
        if row[1] == 'start':
            continue
        # 終了位置で終了
        elif row[1] == 'finish':
            break

        # データの追加
        t_raw.append(float(row[0]))
        y_raw.append(int(row[1]))


plt.figure(figsize=(16, 9))
plt.plot(t_raw, y_raw, 'red', label="Subject")
plt.plot(t_pseudo, y_pseudo, 'blue', linestyle="dashed", label="Pseudo")
plt.xlabel("Time [ms]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
# plt.savefig("pulse.png", bbox_inches='tight', pad_inches=0)

plt.show()
