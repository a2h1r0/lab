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

# x軸の配列を被験者1人あたりの計算結果数(1人あたりのデータ数×学習データ数)分作成
x = list(range(data_size*len(train_file)))
# 計算結果の配列を被験者数分作成
distance = [np.empty(data_size*len(train_file)) for j in range(tester_size)]

# カウンタ用変数の初期化
tester = num = i = 0
min = min*10
max = max*10

for threshold in range(min, max):
    for attack in attack_file:
        for train in train_file:
            dis = np.linalg.norm(attack-train)
            if (dis <= (threshold/10)):
                true += 1
            else:
                false += 1
                
            num += 1

        if (num == (data_size*len(train_file))):
            tester += 1
            
        if ((num/(data_size*len(train_file))) == 1):
            tester += 1
                
        trues[tester][i] = true
        falses[tester][i] = false
    i += 1  # 次の閾値に
    tester = 0
            

# 計算結果を描画
print("正解データは%sで描画" % true_color)
plt.ylabel("distance")
plt.plot(x, distance[0], true_color)      # 1人目(正解)のデータをプロット
for tester in range(1, tester_size):      # 2人目以降のデータをプロット
    r = lambda: random.randint(0, 255)    # グラフの色をランダムに生成
    code = "#{:02X}{:02X}{:02X}"
    color = code.format(r(),r(),r())
    plt.plot(x, distance[tester], color)
plt.show()