import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))


GENERATED_DATA = '20210207_221700_generated'
RAW_DATA = '20210207_221700_raw'
SHOW_EPOCH = 1


t = []
y_generated = []
with open('./data/' + GENERATED_DATA + '.csv') as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        if int(row[0]) < SHOW_EPOCH:
            continue
        elif int(row[0]) == SHOW_EPOCH:
            t.append(float(row[1]) / 1000)
            y_generated.append(int(row[2]))
        elif int(row[0]) > SHOW_EPOCH:
            break

y_raw = []
with open('./data/' + RAW_DATA + '.csv') as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        if int(row[0]) < SHOW_EPOCH:
            continue
        elif int(row[0]) == SHOW_EPOCH:
            y_raw.append(int(row[1]))
        elif int(row[0]) > SHOW_EPOCH:
            break

plt.figure(figsize=(16, 9))
plt.plot(t, y_generated, 'red', label="Generated")
plt.plot(t, y_raw, 'blue', label="Raw")
# plt.ylim(400, 600)
plt.xlabel("Time [s]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
# plt.savefig("../figure/256_generated_1400epoch.png", bbox_inches='tight', pad_inches=0)

plt.show()
