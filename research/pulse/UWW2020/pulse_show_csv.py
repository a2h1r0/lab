import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

t = []
y = []
with open('20201211_213002_ticwatch.csv') as f:

    reader = csv.reader(f)
    next(reader)

    for row in reader:
        t.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(t, y)
plt.show()
