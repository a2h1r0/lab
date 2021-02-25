import serial
import matplotlib.pyplot as plt

t = []
y = []

FINISH = 10

ser = serial.Serial("COM3", 115200)
ser.reset_input_buffer()

while True:
    read_data = ser.readline().rstrip().decode(encoding="utf-8")
    data = read_data.split(",")
    print(data)

    if str.isdecimal(data[0]) and len(data) == 2:
        time = float(data[0])
        if str.isdecimal(data[1]):
            t.append(float(data[0]))
            y.append(float(data[1]))
        else:
            continue
    else:
        continue

    if time >= FINISH*1000000:
        break

plt.plot(t, y)
plt.show()

ser.close()
