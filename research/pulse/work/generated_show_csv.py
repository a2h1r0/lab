import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))


GENERATED_DATA = '20201230_135654_generated.csv'
SHOW_EPOCH = 30


t = []
y = []
with open(GENERATED_DATA) as f:
    reader = csv.reader(f)

    # ヘッダーのスキップ
    next(reader)

    epoch = 0
    for row in reader:
        if epoch != SHOW_EPOCH:
            # 開始位置まで待機
            if row[1] != 'start':
                continue
            # 区切りの検知
            elif row[1] == 'start':
                epoch += 1
                continue
        elif epoch == SHOW_EPOCH:
            # 終了位置で終了
            if row[1] == 'finish':
                break
            # データの追加
            t.append(float(row[0]))
            y.append(int(row[1]))


plt.figure(figsize=(16, 9))
plt.plot(t, y, 'red', label="Generated")
plt.xlabel("Time [ms]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
# plt.savefig("pulse.png", bbox_inches='tight', pad_inches=0)

plt.show()
