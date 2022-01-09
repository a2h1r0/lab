import csv
import serial
import datetime
import os
os.chdir(os.path.dirname(__file__))


GET_TIME = 60  # 取得時間


ser = serial.Serial('COM6', 14400)
ser.reset_input_buffer()
time = 0

now = datetime.datetime.today()
filename = './' + now.strftime('%Y%m%d_%H%M%S') + '_real.csv'

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['timestamp', 'pulse'])

    exit_flag = False
    while True:
        read_data = ser.readline().rstrip().decode(encoding='utf-8', errors='ignore')
        data = read_data.split(',')

        if len(data) == 2 and data[0].isdecimal() and data[1].isdecimal():
            # 異常値の除外（次の値と繋がって，異常な桁数の場合あり）
            if time != 0 and len(str(int(float(data[0]) / 1000000))) > len(str(int(time))) + 2:
                continue

            time = float(data[0]) / 1000
            pulse = int(data[1])

            if time >= GET_TIME * 1000:
                exit_flag = True
                break
            else:
                writer.writerow([time, pulse])
                print([time, pulse])
        else:
            continue

        if exit_flag:
            break

    ser.close()
