import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

t = []
y = []
with open('20200928_212642_surface.csv') as f:

    reader = csv.reader(f)
    next(reader)
    next(reader)

    for row in reader:
        t.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(t, y)
plt.show()
