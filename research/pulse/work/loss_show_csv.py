import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import os
os.chdir(os.path.dirname(__file__))

LOSS_DATA = '20210126_235256_loss.csv'


epoch = []
D_loss = []
G_loss = []
with open(LOSS_DATA) as f:
    reader = csv.reader(f)

    # ヘッダーのスキップ
    next(reader)

    for row in reader:
        # データの追加
        epoch.append(float(row[0]))
        D_loss.append(float(row[1]))
        G_loss.append(float(row[2]))


plt.figure(figsize=(16, 9))
plt.plot(epoch, D_loss, 'red', label="Discriminator")
plt.plot(epoch, G_loss, 'blue', label="Generator")
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.savefig("../figure/256_generated_1400epoch_loss.png",
            bbox_inches='tight', pad_inches=0)

plt.show()
