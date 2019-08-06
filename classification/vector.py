import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
attack = np.loadtxt('exam_attack.csv', delimiter = ",", dtype = float, skiprows = 1, usecols = range(0,20))
train = np.loadtxt('exam_train.csv', delimiter = ",", dtype = float, skiprows = 1, usecols = range(0,20))
i = 0
height = np.empty(len(attack)*len(train))

for vec in attack:
    for vec2 in train:
        height[i] = np.linalg.norm(vec-vec2)
        i+=1

left = list(range(i))


plt.bar(left, height, width=1.0)