import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

file1 = '20200926_233009_fujii.csv'
file2 = '20200928_212642_surface.csv'

t1 = []
y1 = []
with open(file1) as f:

    reader = csv.reader(f)
    next(reader)
    next(reader)

    for row in reader:
        if float(row[0]) < 2:
            continue
        elif 8 < float(row[0]):
            break

        t1.append(float(row[0])*1000)
        y1.append(float(row[1]))

t2 = []
y2 = []
with open(file2) as f:

    reader = csv.reader(f)
    next(reader)
    next(reader)

    for row in reader:
        if float(row[0]) < 2:
            continue
        elif 8 < float(row[0]):
            break

        t2.append(float(row[0])*1000)
        y2.append(float(row[1]))


plt.figure(figsize=(16, 9))
plt.plot(t1, y1, 'red', label="Subject")
plt.plot(t2, y2, 'blue', linestyle="dashed", label="Pseudo")
plt.xlabel("Time [ms]", fontsize=18)
plt.ylabel("Pulse sensor value", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.savefig("pulse.png", bbox_inches='tight', pad_inches=0)

plt.show()
