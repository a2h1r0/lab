import csv
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.path.dirname(__file__))


REAL_DATA = './20220109_163329_real_rr.csv'
GENERATED_DATA = './20220110_052117_generated_rr.csv'


real_time, real_rr = [], []
with open(REAL_DATA) as f:
    reader = csv.reader(f)

    for row in reader:
        real_time.append(float(row[0]))
        real_rr.append(float(row[1]))

generated_time, generated_rr = [], []
with open(GENERATED_DATA) as f:
    reader = csv.reader(f)

    for row in reader:
        generated_time.append(float(row[0]))
        generated_rr.append(float(row[1]))


plt.figure(figsize=(16, 4))
plt.plot(real_time, real_rr, label='Real', color='blue')
plt.plot(generated_time, generated_rr, label='Generated', color='red', linestyle='dashed')
plt.xlabel('Time [s]', fontsize=26)
plt.ylabel('RR interval [s]', fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26, loc='lower right')

plt.savefig('../figures/rr_wave.eps', bbox_inches='tight', pad_inches=0)

plt.show()
