import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet

## データの読み込み ##
train = pd.read_csv('train.csv', usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF"], encoding='Shift-JIS')

attack = pd.read_csv('attack.csv', usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF"], encoding='Shift-JIS')

# MCD
mcd = MinCovDet()
mcd.fit(train)
anomaly_score_mcd = mcd.mahalanobis(attack)

# 最尤法
mle = EmpiricalCovariance()
mle.fit(train)
anomaly_score_mle = mle.mahalanobis(attack)

x = range(83)

plt.figure(0)
plt.scatter(x,anomaly_score_mcd)
plt.figure(1)
plt.scatter(x,anomaly_score_mcd)
plt.show()