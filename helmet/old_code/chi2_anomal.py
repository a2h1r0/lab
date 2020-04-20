import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from statistics import mean, variance
from scipy import stats

test = pd.read_csv('testdata.csv', usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF"], encoding='Shift-JIS')

data = test[78:100]
data= data.values.tolist()

# 標本平均
for i in range(22):
    data[i] = mean(data[i])
mean = mean(data)

# 標本分散
variance = variance(data)

# 異常度
anomaly_scores = []
for x in data:
    anomaly_score = (x - mean)**2 / variance
    anomaly_scores.append(anomaly_score)

# カイ二乗分布による1%水準の閾値
threshold = stats.chi2.interval(0.99, 1)[1]

num = range(22)
# 結果の描画
plt.plot(num, anomaly_scores, "o", color = "b")
plt.plot([0,22],[threshold, threshold], 'k-', color = "r", ls = "dashed")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.ylim([0,100])
plt.show()