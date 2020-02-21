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



file_num = ['3', '11', '21']
directory = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']


# take_set[身体部位][ファイル番号*被験者1~3]

take_data = [[] for i in range(len(directory))]
for part, part_name in enumerate(directory):
    for num in file_num:
        path = glob.glob('.\\'+part_name+'\\subject?_file_'+num+'.csv')
        for filename in path:
            if os.path.isfile(filename):
                data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2]) 
                if data != []:
                    take_data[part].append([0 for i in range(6)])
                    take_data[part][-1] = list(np.concatenate([np.mean(data, axis=0), np.var(data, axis=0)]))
                    
        if take_set[part] == []:
            del take_set[part]             

# これで部位ごとに学習にかける．
label = []
for i in range(directory):
    label.append(['take' for i in range(take_set)])
clf = svm.SVC(C=1.0, kernel='linear')
cross_val_score(clf, tale_data[part], label, cv=5)
print('Cross-Validation scores: {}\n'.format(scores[-1]))
        
            
                
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