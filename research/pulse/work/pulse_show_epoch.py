import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

EPOCH = 100  # 描画するエポック数（10刻み）

GENERATED_DATA = '20210124_234351_generated2.csv'


t_raw = []
y_raw = []
t_generated = []
y_generated = []
with open(GENERATED_DATA) as f:
    reader = csv.reader(f)

    for index in range((EPOCH // 10) - 1):
        next(reader)
        next(reader)

    for index, row in enumerate(reader):
        if index == 0:
            t_raw = list(range(len(row)))
            y_raw = list(map(float, row))
        if index == 1:
            t_generated = list(range(len(row)))
            y_generated = list(map(float, row))


plt.figure(figsize=(16, 9))
plt.plot(t_raw, y_raw, 'red', label="Subject")
plt.plot(t_generated, y_generated, 'blue',
         linestyle="dashed", label="generated")
# plt.xlabel("Time [ms]", fontsize=18)
# plt.ylabel("Pulse sensor value", fontsize=18)
# plt.tick_params(labelbottom=False,
#                 labelleft=False,
#                 labelright=False,
#                 labeltop=False)
# plt.legend(fontsize=18, loc='upper right')
# plt.savefig("pulse.png", bbox_inches='tight', pad_inches=0)

plt.show()
