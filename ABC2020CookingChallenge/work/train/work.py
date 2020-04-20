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
warnings.simplefilter('ignore')


sand_take = ['subject2_file_457', 'subject2_file_646', 'subject2_file_743', 'subject2_file_308', 'subject2_file_325','subject2_file_540', 'subject3_file_867', 'subject3_file_850', 'subject3_file_97', 'subject3_file_554', 'subject3_file_34', 'subject3_file_264', 'subject3_file_808', 'subject3_file_216', 'subject1_file_622', 'subject1_file_869', 'subject1_file_588', 'subject1_file_140', 'subject1_file_808', 'subject1_file_238', 'subject1_file_618', 'subject1_file_964', 'subject1_file_884', 'subject1_file_243', 'subject1_file_957']

fruit_take = ['subject2_file_138', 'subject2_file_452', 'subject2_file_923', 'subject2_file_49', 'subject2_file_588', 'subject2_file_171', 'subject3_file_382', 'subject3_file_817', 'subject3_file_5', 'subject3_file_9', 'subject3_file_531', 'subject1_file_873', 'subject1_file_37']

sand_put = ['subject2_file_368', 'subject2_file_101', 'subject2_file_180', 'subject2_file_586', 'subject2_file_565', 'subject2_file_1', 'subject2_file_581', 'subject3_file_718', 'subject3_file_36', 'subject3_file_146', 'subject3_file_104', 'subject3_file_935']

sand_put_other = ['subject1_file_830', 'subject1_file_272', 'subject1_file_389', 'subject1_file_247', 'subject1_file_980', 'subject1_file_310', 'subject1_file_98', 'subject1_file_424', 'subject1_file_684', 'subject1_file_968', 'subject1_file_11', 'subject1_file_814']

sand_wash_take = ['subject2_file_679', 'subject2_file_52', 'subject2_file_485', 'subject2_file_263', 'subject2_file_792', 'subject2_file_635', 'subject3_file_537', 'subject3_file_901', 'subject3_file_389', 'subject3_file_829', 'subject1_file_780', 'subject1_file_319']

take_files = sand_take + fruit_take
wash_take = sand_wash_take
put_files = sand_put

#file_name = take_files + put_files

directory = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']

# take, put, peel, 
# take[身体部位][ファイル番号*被験者1~3]
take = [[] for i in range(len(directory))]
take_label = [[] for i in range(len(directory))]
for part, part_name in enumerate(directory):
    for file in take_files:
        path = ('./'+part_name+'/'+file+'.csv')
        if os.path.isfile(path):
            data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[0,1,2]) 
            # データの追加
            if data != []:
                take[part].append([0 for i in range(6)])
                take[part][-1] = list(np.concatenate([np.mean(data, axis=0), np.var(data, axis=0)]))
                # ラベルの付与
                take_label[part].append('take')
        # 空なら削除
        if take[part] == []:
            del take[part]
            del take_label[part]


# put[身体部位][ファイル番号*被験者1~3]
put = [[] for i in range(len(directory))]
put_label = [[] for i in range(len(directory))]
for part, part_name in enumerate(directory):
    for file in put_files:
        path = ('./'+part_name+'/'+file+'.csv')
        if os.path.isfile(path):
            data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[0,1,2]) 
            # データの追加
            if data != []:
                put[part].append([0 for i in range(6)])
                put[part][-1] = list(np.concatenate([np.mean(data, axis=0), np.var(data, axis=0)]))
                # ラベルの付与
                put_label[part].append('put')
        # 空なら削除
        if put[part] == []:
            del put[part]
            del put_label[part]
            


left_hip = take[0] + put[0]
left_hip_label = take_label[0] + put_label[0]

right_arm = take[1] + put[1]
right_arm_label = take_label[1] + put_label[1]

left_wrist = take[2] + put[2]
left_wrist_label = take_label[2] + put_label[2]

right_wrist = take[3] + put[3]
right_wrist_label = take_label[3] + put_label[3]

 

# これで部位ごとに学習にかける．
clf = svm.SVC(C=1.0, kernel='linear')

score = cross_val_score(clf, left_hip, left_hip_label, cv=5)
print('\nleft_hip'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')
        
score = cross_val_score(clf, right_arm, right_arm_label, cv=5)
print('\nright_arm'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')

score = cross_val_score(clf, left_wrist, left_wrist_label, cv=5)
print('\nleft_wrist'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')

score = cross_val_score(clf, right_wrist, right_wrist_label, cv=5)
print('\nright_wrist'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')

            


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
# take[身体部位][ファイル番号*被験者1~3]
take_wash = [[] for i in range(len(directory))]
take_wash_label = [[] for i in range(len(directory))]
for part, part_name in enumerate(directory):
    for file in take_files:
        path = ('./'+part_name+'/'+file+'.csv')
        if os.path.isfile(path):
            data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=[0,1,2]) 
            # データの追加
            if data != []:
                take_wash[part].append([0 for i in range(6)])
                take_wash[part][-1] = list(np.concatenate([np.mean(data, axis=0), np.var(data, axis=0)]))
                # ラベルの付与
                take_wash_label[part].append('take')
        # 空なら削除
        if take_wash[part] == []:
            del take_wash[part]
            del take_wash_label[part]
    

# OneClassSVM：ダメダメ

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

clf.fit(take[0])
pred = clf.predict(take_wash[0])
print('\n[0]: ')
print(pred)
clf.fit(take[1])
pred = clf.predict(take_wash[1])
print('\n[1]: ')
print(pred)
clf.fit(take[2])
pred = clf.predict(take_wash[2])
print('\n[2]: ')
print(pred)
clf.fit(take[3])
pred = clf.predict(take_wash[3])
print('\n[3]: ')
print(pred)
    



# LOF：ダメダメ

clf = LocalOutlierFactor(n_neighbors=3,
                           novelty=True,
                           contamination=0.1)

clf.fit(take[0])
pred = clf.predict(take_wash[0])
print('\n[0]: ')
print(pred)
clf.fit(take[1])
pred = clf.predict(take_wash[1])
print('\n[1]: ')
print(pred)
clf.fit(take[2])
pred = clf.predict(take_wash[2])
print('\n[2]: ')
print(pred)
clf.fit(take[3])
pred = clf.predict(take_wash[3])
print('\n[3]: ')
print(pred)
"""
