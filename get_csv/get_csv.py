import serial
from time import sleep
import os.path
import csv

ser = serial.Serial('COM5', 9600)   # Arduino1号機
ser2 = serial.Serial('COM6', 9600)  # Arduino2号機
filename = 'train.csv'               # データ保存先ファイル
data_size = 100                      # データ取得回数
tester = "Fujii"                     # 正解ラベル(被験者名)

sleep(1)    # ポート準備に1秒待機**これがないとシリアル通信がうまく動かない**

exist = 0                       # 分岐処理用
if os.path.isfile(filename):    # データ保存ファイルの存在確認
    exist = 1
    
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)

    if exist == 0:      # ファイルが新規作成の場合，ラベルを付与する
        writer.writerow(["in0","in1","in2","in3","in4","in5","in6","in7","in8",
                         "in9","inA","inB","inC","inあ","inい","inう",
                         "in10","in11","in12","in13","in14","in15","in16",
                         "in17","in18","in19","inD","inE","inF","inア","inイ",
                         "inウ","Tester"])
    elif exist == 1:    # ファイルが既存の場合，ラベルを付与しない
        print("File exist. Didn't give label.")
   
    # データの取得と書き込み
    for row in range(data_size):
        # シリアル通信とデータ加工
        ser.write("!".encode('UTF-8'))
        voltage = ser.readline().decode('UTF-8').rstrip().split()
        ser2.write("?".encode('UTF-8'))
        voltage2 = ser2.readline().decode('UTF-8').rstrip().split()

        voltage.extend(voltage2)    # 1号機と2号機のデータを結合
        voltage.append(tester)      # 最終列に正解ラベル(被験者名)を追加
        
        writer.writerow(voltage)    # ファイルへ書き込み

ser.close()
ser2.close()
print("Finish")