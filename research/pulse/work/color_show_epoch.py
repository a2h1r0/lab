import matplotlib.pyplot as plt
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


DATA = './data/20210214_210139/colors_7000.csv'
SHOW_EPOCH = 100

# データの読み出し
y_real = []
y_fake = []
with open(DATA) as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        if int(row[0]) < SHOW_EPOCH:
            continue
        elif int(row[0]) == SHOW_EPOCH:
            y_real.append(int(row[1]))
            y_fake.append(int(row[2]))
        elif int(row[0]) > SHOW_EPOCH:
            break


plt.figure(figsize=(16, 9))
plt.plot(list(range(len(y_real))), y_real, 'blue', label='Real')
plt.plot(list(range(len(y_fake))), y_fake, 'red', label='Fake')
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('Pulse sensor value', fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')

plt.show()
