import pandas as pd
#from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# CSVファイルの読み込み
train = pd.read_csv('train.csv', usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF"], encoding='Shift-JIS') # 学習データ
test = pd.read_csv('attack.csv', usecols=["in0","in1","in2","in3","in4","in5","in6","in7",
                                            "in8","in9","inあ","inい","inう","inA","inB","inC",
                                            "in10","in11","in12","in13","in14","in15","in16",
                                            "in17","in18","in19","inア","inイ","inウ","inD","inE",
                                            "inF"], encoding='Shift-JIS')   # 評価データ


clf = LocalOutlierFactor(n_neighbors=5)
pred = clf.fit_predict(train)
"""
clf = OneClassSVM(nu=0.2, kernel='rbf', gamma=1/32)
clf.fit(train)
pred = clf.predict(test)
"""