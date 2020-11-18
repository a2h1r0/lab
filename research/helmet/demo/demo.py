## 編集箇所 ##
import sys
import os
import os.path
import serial
from time import sleep
import numpy as np
from calculate_vector_ave_cols import calculate_vector_ave_cols as cal
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import itertools
import copy


USB_PORT1 = 'COM5'
USB_PORT2 = 'COM4'


time = 2    # **データ取得時間(秒)**
## ここまで ##

sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
#from make_traindata import make_traindata as make


tester = ['ooyama', 'okamoto', 'kajiwara', 'sawano',
          'nagamatsu', 'noda', 'hatta', 'fujii', 'matsuda']  # **被験者**


sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
# 線形SVMのインスタンスを生成
clf = svm.SVC(kernel='linear', random_state=None)

cols = ['in0', 'in1', 'in2', 'in3', 'in4', 'in5', 'in6', 'in7',
        'in8', 'in9', 'inあ', 'inい', 'inう', 'inA', 'inB', 'inC',
        'in10', 'in11', 'in12', 'in13', 'in14', 'in15', 'in16',
        'in17', 'in18', 'in19', 'inア', 'inイ', 'inウ', 'inD', 'inE',
        'inF', 'Number']


def make_dataset():
    data = []
    label = []
    for item in vector_ave:
        data.extend(item)
    for index, name in enumerate(tester):
        label.extend([name]*len(vector_ave[index]))
    return data, label


vector_ave = cal(tester, cols)  # ベクトルの平均値を計算
data, label = make_dataset()

# モデルの学習。fit関数で行う。
clf.fit(data, label)

test = [3.80981818, 4.79145455, 4.99, 4.99, 4.47254545, 4.98672727, 4.99, 4.99,
        4.99, 4.99, 4.99, 2.29054545, 4.548, 3.47072727, 3.49909091, 3.27327273, 4.42781818, 4.99,
        4.83727273, 4.99, 4.4245454, 3.65763636, 4.02236364, 4.266, 4.99, 4.99, 4.99, 2.44563636,
        4.99, 3.8352727, 3.19890909, 3.22927273]
test = np.array(test).reshape(1, -1)


# Arduinoの用意
ser = serial.Serial(USB_PORT1, 57600)
ser2 = serial.Serial(USB_PORT2, 57600)
sleep(1)    # ポート準備に1秒待機**これがないとシリアル通信がうまく動かない**


def voltage_get():
    # シリアル通信
    ser.write(bytes('1'+'\0', 'utf-8'))
    voltage = ser.readline().decode('UTF-8').rstrip().split()
    ser2.write(bytes('1'+'\0', 'utf-8'))
    voltage2 = ser2.readline().decode('UTF-8').rstrip().split()

    # データ整列
    voltage.extend(voltage2)                    # 1号機と2号機のデータを結合
    voltage = list(map(float, voltage))

    return voltage


## Arduinoと接続，再実行時はここまで省略 ##
print("\n\n--------- 開始します ---------\n\n")
ser2.write(bytes('0'+'\0', 'utf-8'))

# データの取得と書き込み
while True:
    if sum(voltage_get()) > 145:
        print("ヘルメットを装着してください．\n\n")
        ser2.write(bytes('0'+'\0', 'utf-8'))
        while True:
            voltage = voltage_get()
            if sum(voltage) < 145 or voltage[12] < 1:
                sleep(3)
                break

    else:
        # ここで機械学習

        voltage = np.array(voltage).reshape(1, -1)
        pred_test = clf.predict(voltage)

        print("あなたは", end="")
        print(pred_test[0], end="")
        print("さんですね！\n\n")
        ser2.write(bytes('2'+pred_test[0]+'\0', 'utf-8'))
        while True:
            voltage = voltage_get()
            if sum(voltage) > 145 or voltage[12] < 1:
                break

    if voltage[12] < 1:
        break


print("Finish\n")
ser2.write(bytes('3'+'\0', 'utf-8'))

# Arduinoの終了
ser.close()
ser2.close()
