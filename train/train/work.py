import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy import signal
from numpy.fft import fft, ifft, fftfreq
from sklearn.cluster import KMeans
from fastdtw import fastdtw
import statsmodels.api as sm # 季節成分分解
import seaborn as sns
import glob
import re
import os
import warnings
import sys
from sklearn import svm
from sklearn.model_selection import cross_val_score


sand_take = ['subject2_file_457', 'subject2_file_646', 'subject2_file_743', 'subject2_file_308', 'subject2_file_325','subject2_file_540', 'subject3_file_867', 'subject3_file_850', 'subject3_file_97', 'subject3_file_554', 'subject3_file_34', 'subject3_file_264', 'subject3_file_808', 'subject3_file_216', 'subject1_file_622', 'subject1_file_869', 'subject1_file_588', 'subject1_file_140', 'subject1_file_808', 'subject1_file_238', 'subject1_file_618', 'subject1_file_964', 'subject1_file_884', 'subject1_file_243', 'subject1_file_957']

sand_put = ['subject2_file_368', 'subject2_file_101', 'subject2_file_180', 'subject2_file_586', 'subject2_file_565', 'subject2_file_1', 'subject2_file_581', 'subject3_file_718', 'subject3_file_36', 'subject3_file_146', 'subject3_file_104', 'subject3_file_935']

sand_put_other = ['subject1_file_830', 'subject1_file_272', 'subject1_file_389', 'subject1_file_247', 'subject1_file_980', 'subject1_file_310', 'subject1_file_98', 'subject1_file_424', 'subject1_file_684', 'subject1_file_968', 'subject1_file_11', 'subject1_file_814']

fruit_take = ['subject2_file_138', 'subject2_file_452', 'subject2_file_923', 'subject2_file_49', 'subject2_file_588', 'subject2_file_171', 'subject3_file_382', 'subject3_file_817', 'subject3_file_5', 'subject3_file_9', 'subject3_file_531', 'subject1_file_873', 'subject1_file_37']


file_name = sand_take + sand_put + fruit_take

directory = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']


# take_set[身体部位][ファイル番号*被験者1~3]
take_set = [[] for i in range(len(directory))]
label = [[] for i in range(len(directory))]
for part, part_name in enumerate(directory):
    for file in file_name:
        path = ('./'+part_name+'/'+file+'.csv')
        if os.path.isfile(path):
            data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=[0,1,2]) 
            # データの追加
            if data != []:
                take_set[part].append([0 for i in range(6)])
                take_set[part][-1] = list(np.concatenate([np.mean(data, axis=0), np.var(data, axis=0)]))
                # ラベルの付与
                if len(take_set[part]) <= len(sand_take):
                    label[part].append('sand_take')
                elif len(sand_take) < len(take_set[part]) and len(take_set[part]) <= (len(sand_take)+len(sand_put)):
                    label[part].append('sand_put')
                else:
                    label[part].append('fruit_take')

        # 空なら削除
        if take_set[part] == []:
            del take_set[part]
            del label[part]
    



# これで部位ごとに学習にかける．
clf = svm.SVC(C=1.0, kernel='linear')
for part, part_name in enumerate(directory):
    if part == 2:
        continue
    score = cross_val_score(clf, take_set[part], label[part], cv=5)
    print('\n部位 : '+part_name+'\nCross-Validation scores: {}\n'.format(np.mean(score)))
        
            


"""
K-Means

kmeans_model = KMeans(n_clusters=2, random_state=10).fit(left_hip_test)
labels = kmeans_model.labels_
print(labels)
"""



"""
# PCA：いらんかった

pca = PCA(n_components=2)
pca.fit(train)
pca_train = pca.components_
pca.fit(test)
pca_test = pca.components_

x_axis = list(range(len(train)))
plt.scatter(x_axis, pca_train)
plt.scatter(x_axis, pca_test)   
"""
    

"""
# OneClassSVM：ダメダメ

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(train)
pred = clf.predict(test)
print(pred)
"""    


"""
# LOF：ダメダメ

model = LocalOutlierFactor(n_neighbors=1,
                           novelty=True,
                           contamination=0.1)
model.fit(train) # train_dataは正常データが大多数であるような訓練データ
prediction = model.predict(test) # テストデータに対する予測
print(prediction)
"""