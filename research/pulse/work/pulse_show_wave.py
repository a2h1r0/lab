import serial
import matplotlib.pyplot as plt

t = []
y = []

FINISH = 10

ser = serial.Serial("COM3", 115200)
ser.reset_input_buffer()

while(1):
    read_data = ser.readline().rstrip().decode(encoding="utf-8")
    devide_data = read_data.split(",")
    print(devide_data)

    if(str.isdecimal(devide_data[0]) and len(devide_data) == 2):
        ard_time = float(devide_data[0])
        if str.isdecimal(devide_data[1]):
            t.append(float(devide_data[0]))
            y.append(float(devide_data[1]))
        else:
            continue
    else:
        continue

    if(ard_time >= FINISH*1000000):
        break

plt.plot(t, y)
plt.show()

ser.close()
