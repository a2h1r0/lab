import serial
import matplotlib.pyplot as plt
from scipy import signal

t = []
y = []

FINISH = 10

ser = serial.Serial("COM3", 9600)
ser.reset_input_buffer()

while True:
    read_data = ser.readline().rstrip().decode(encoding="utf-8")
    data = read_data.split(",")
    print(data)

    if str.isdecimal(data[0]) and len(data) == 2:
        time = float(data[0])
        if str.isdecimal(data[1]):
            t.append(float(data[0]))
            y.append(int(data[1]))
        else:
            continue
    else:
        continue

    if time >= FINISH*1000000:
        break

peaks, _ = signal.find_peaks(y, distance=30)
plt.plot(t, y)
for p in peaks:
    plt.plot(t[p], y[p], 'ro')
plt.show()

ser.close()
