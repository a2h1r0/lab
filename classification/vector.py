import numpy as np
import random
import matplotlib.pyplot as plt

data_size = 500
tester_size = 9
true_color = 'red'

# CSVファイルの読み込み
attack = np.loadtxt('vector_attack.csv', delimiter = ",", dtype = float, skiprows = 1, usecols = range(0,20))
train = np.loadtxt('vector_train.csv', delimiter = ",", dtype = float, skiprows = 1, usecols = range(0,20))
#
# 計算結果数の配列(1人あたりのデータ数×学習データ数)を被験者数分作成
distance = [np.empty(data_size*len(train)) for j in range(tester_size)]

i=0
line=-1

for vec in attack:
    if (i%(data_size*len(train)) == 0):
        line+=1
        i=0
    for vec2 in train:
        distance[line][i] = np.linalg.norm(vec-vec2)
        i+=1

x = list(range(data_size*len(train)))

print("正解データは%sで描画" % true_color)
plt.plot(x, distance[0], true_color)
for j in range(1, 9):
    r = lambda: random.randint(0,255)
    code = "#{:02X}{:02X}{:02X}"
    color = code.format(r(),r(),r())
    plt.plot(x, distance[j], color)

plt.show()