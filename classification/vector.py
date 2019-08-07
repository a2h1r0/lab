import numpy as np
import random
import matplotlib.pyplot as plt

data_size = 500     # 被験者1人あたりのデータ数
tester_size = 9     # 被験者数
true_color = 'red'  # 正解データのグラフ色

# CSVファイルの読み込み，認証データ1人目には学習データに用いた被験者の別データを格納
attack_file = np.loadtxt('vector_attack.csv', delimiter = ",", dtype = float, 
                    skiprows = 1, usecols = range(0,20)) # 認証データ
train_file = np.loadtxt('vector_train.csv', delimiter = ",", dtype = float, 
                   skiprows = 1, usecols = range(0,20))  # 学習データ

# x軸の配列宣言，被験者1人あたりの計算結果数(1人あたりのデータ数×学習データ数)
x = list(range(data_size*len(train_file)))
# 計算結果数の配列を被験者数分作成
distance = [np.empty(data_size*len(train_file)) for j in range(tester_size)]

# カウンタ用変数の初期化
tester = num = 0

# ベクトル距離計算
for attack in attack_file:
    # 認証データ1つにつき，学習データ全てに対して距離計算
    for train in train_file:
        distance[tester][num] = np.linalg.norm(attack-train)
        num += 1
    # 被験者と配列の切り替え
    # 被験者1人あたりのデータ数×学習データ数回計算が終われば，次の被験者へ移る
    if (num%(data_size*len(train_file)) == 0):
        tester += 1
        num = 0

# 計算結果を描画
print("正解データは%sで描画" % true_color)
plt.plot(x, distance[0], true_color)        # 1人目(学習用の被験者)のデータをプロット
for tester in range(1, tester_size):
    r = lambda: random.randint(0,255)       # グラフの色をランダムに生成
    code = "#{:02X}{:02X}{:02X}"
    color = code.format(r(),r(),r())
    plt.plot(x, distance[tester], color)    # 2人目以降のデータをプロット
plt.show()