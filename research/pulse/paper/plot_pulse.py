import csv
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.path.dirname(__file__))


REAL_DATA = './raw.csv'
GENERATED_DATA = './generate.csv'
END = 5  # 表示終了時間（秒）


real_time, real_pulse = [], []
with open(REAL_DATA) as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        time = float(row[0]) / 1000
        if time > END:
            break

        real_time.append(time)
        real_pulse.append(int(row[1]))

generated_time, generated_pulse = [], []
with open(GENERATED_DATA) as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        time = float(row[0]) / 1000
        if time > END:
            break

        generated_time.append(time)
        generated_pulse.append(int(row[1]))


plt.figure(figsize=(16, 9))
plt.plot(real_time, real_pulse, label='Real', color='blue')
plt.plot(generated_time, generated_pulse, label='Generated', color='red', linestyle='dashed')
plt.xlabel('Time [s]', fontsize=26)
plt.ylabel('Pulse Value', fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26, loc='upper right')

plt.savefig('../figures/pulse_wave.eps', bbox_inches='tight', pad_inches=0)

plt.show()
