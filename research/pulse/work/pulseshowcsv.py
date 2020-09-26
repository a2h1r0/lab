import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

num = 10  # 移動平均の個数

t = []
pulse_value = []
with open('20200926_233009_fujii.csv') as f:

    reader = csv.reader(f)
    next(reader)
    next(reader)

    for row in reader:
        t.append(float(row[0]))
        pulse_value.append(float(row[1]))

y = np.ones(num)/num
y1 = np.convolve(pulse_value, y, mode='same')  # 移動平均
# maxid = signal.argrelmax(pulse_value, order=10)

plt.plot(t[5:-5], y1[5:-5])
plt.show()
