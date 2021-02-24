import matplotlib.pyplot as plt
import csv
from natsort import natsorted
import glob
import os
os.chdir(os.path.dirname(__file__))


TIME = '20210224_160755'
SHOW_EPOCH = 5000


t = []
y_generated = []
files = natsorted(glob.glob('./data/' + TIME + '/generated_*.csv'))
for data in files:
    with open(data) as f:
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

        if len(y_generated) > 0:
            break

y_raw = []
files = natsorted(glob.glob('./data/' + TIME + '/raw_*.csv'))
for data in files:
    with open(data) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if int(row[0]) < SHOW_EPOCH:
                continue
            elif int(row[0]) == SHOW_EPOCH:
                y_raw.append(int(row[1]))
            elif int(row[0]) > SHOW_EPOCH:
                break

        if len(y_raw) > 0:
            break


plt.figure(figsize=(16, 9))
plt.plot(t, y_generated, 'red', label='Generated')
plt.plot(t, y_raw, 'blue', label='Raw')
# plt.ylim(400, 600)
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('Pulse sensor value', fontsize=18)
plt.title('Epoch: ' + str(SHOW_EPOCH))
plt.tick_params(labelsize=18)
plt.legend(fontsize=18, loc='upper right')
plt.savefig('../figure/10000_generated_' + str(SHOW_EPOCH) + 'epoch.png',
            bbox_inches='tight', pad_inches=0)

plt.show()
