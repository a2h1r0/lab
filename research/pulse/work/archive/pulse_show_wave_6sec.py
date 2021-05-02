import serial
import matplotlib.pyplot as plt

show_data = []
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
            show_data.append(data)
        else:
            continue
    else:
        continue

    if time >= FINISH*1000000:
        break

for row in show_data:
    if float(row[0]) < 2 * 1000000:
        continue
    elif 8 * 1000000 < float(row[0]):
        break

    t.append(float(row[0]))
    y.append(float(row[1]))

plt.plot(t, y)
plt.show()

ser.close()
