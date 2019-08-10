import numpy as np
import matplotlib.pyplot as plt
import random

data_size = 500     # 被験者1人あたりのデータ数
tester_size = 9     # 被験者数
true_color = 'red'  # 正解データのグラフ色

min = 1.5
max = 2.5

# CSVファイルの読み込み，認証データ1人目には学習データに用いた被験者の別データを正解として格納
attack_file = np.loadtxt('vector_attack.csv', delimiter = ",", dtype = float, 
                    skiprows = 1, usecols = range(0,20)) # 認証データ
train_file = np.loadtxt('vector_train.csv', delimiter = ",", dtype = float, 
                   skiprows = 1, usecols = range(0,20))  # 学習データ

x = np.arange(min, max, 0.1)

# カウンタ用変数の初期化
tester = loop = threshold = 0
min = int(min*10)
max = int(max*10)

TAN = np.empty(max-min) # True Acceptance Number(本人受入数)
FAN = np.empty(max-min) # False Acceptance Number(他人受入数)
TRN = np.empty(max-min) # True Rejection Number(他人拒否数)
FRN = np.empty(max-min) # False Rejection Number(本人拒否数)


for thresholds in range(min, max):
    for attack in attack_file:
        for train in train_file:
            distance = np.linalg.norm(attack-train)
            
            if (distance <= (thresholds/10)):
                if (loop < (data_size*len(train_file))):
                    TAN[threshold] += 1
                else:
                    FAN[threshold] += 1
                    
            elif (distance > (thresholds/10)):
                if (loop < (data_size*len(train_file))):
                    FRN[threshold] += 1
                else:
                    TRN[threshold] += 1
            loop += 1
            
    threshold += 1  # 次の閾値に
    loop = 0
            
for i in range(threshold):
    TAN[i] = TAN[i]/(data_size*len(train_file))
    FAN[i] = TAN[i]/(data_size*len(train_file)*(tester_size-1))
    TRN[i] = TAN[i]/(data_size*len(train_file)*(tester_size-1))
    FRN[i] = TAN[i]/(data_size*len(train_file))

    # 計算結果を描画
#plt.xlim(min/10, max/10)
#plt.ylim(0, (data_size*len(train_file)))
plt.xlabel("threshold")
plt.plot(x, TAN, 'r', label="True Acceptance Number")      # 1人目(正解)のデータをプロット
plt.plot(x, FAN, 'g', label="False Acceptance Number")
plt.plot(x, FRN, 'b', label="True Rejection Number")
plt.plot(x, TRN, 'y', label="False Rejection Number")
plt.legend()

#for tester in range(1, tester_size):      # 2人目以降のデータをプロット
#    r = lambda: random.randint(0, 255)    # グラフの色をランダムに生成
#    code = "#{:02X}{:02X}{:02X}"
#    color = code.format(r(),r(),r())
#    plt.plot(x, distance[tester], color)
plt.show()