import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))


GENERATED_DATA = '20210126_235256_generated.csv'
SHOW_EPOCH = 500


t = []
y = []
with open(GENERATED_DATA) as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        if int(row[0]) != SHOW_EPOCH:
            continue
        elif int(row[0]) == SHOW_EPOCH:
            t.append(float(row[1]) / 1000)
            y.append(int(row[2]))


plt.figure(figsize=(16, 9))
plt.plot(t, y, 'red', label="Generated")
plt.xlabel("Time [s]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
# plt.savefig("../figure/256_generated_1400epoch.png", bbox_inches='tight', pad_inches=0)

plt.show()
