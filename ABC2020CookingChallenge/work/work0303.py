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

clf = svm.SVC(C=1.0, kernel='linear')




# 動作のみ

## データの読み込み ##
filename = 'left_hip.csv'
left_hip_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
left_hip = left_hip_df.values[:, 2:8]
left_hip_label = left_hip_df.values[:, 1:2]

filename = 'right_arm.csv'
right_arm_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_arm = right_arm_df.values[:, 2:8]
right_arm_label = right_arm_df.values[:, 1:2]

filename = 'left_wrist.csv'
left_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
left_wrist = left_wrist_df.values[:, 2:8]
left_wrist_label = left_wrist_df.values[:, 1:2]

filename = 'right_wrist.csv'
right_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_wrist = right_wrist_df.values[:, 2:8]
right_wrist_label = right_wrist_df.values[:, 1:2]


# これで部位ごとに学習にかける．

print('\n\n****動作のみ****')

score = cross_val_score(clf, left_hip, left_hip_label, cv=5)
print('left_hip'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')
        
score = cross_val_score(clf, right_arm, right_arm_label, cv=5)
print('\nright_arm'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')

score = cross_val_score(clf, left_wrist, left_wrist_label, cv=5)
print('\nleft_wrist'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')

score = cross_val_score(clf, right_wrist, right_wrist_label, cv=5)
print('\nright_wrist'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')





# レシピ+動作

## データの読み込み ##
filename = 'left_hip.csv'
left_hip_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
left_hip = left_hip_df.values[:, 2:8]
label = left_hip_df.values[:, 0:2]
left_hip_label = []
for string in label:
    left_hip_label.append('-'.join(string))

filename = 'right_arm.csv'
right_arm_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_arm = right_arm_df.values[:, 2:8]
label = right_arm_df.values[:, 0:2]
right_arm_label = []
for string in label:
    right_arm_label.append('-'.join(string))

filename = 'left_wrist.csv'
left_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
left_wrist = left_wrist_df.values[:, 2:8]
label = left_wrist_df.values[:, 0:2]
left_wrist_label = []
for string in label:
    left_wrist_label.append('-'.join(string))

filename = 'right_wrist.csv'
right_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_wrist = right_wrist_df.values[:, 2:8]
label = right_wrist_df.values[:, 0:2]
right_wrist_label = []
for string in label:
    right_wrist_label.append('-'.join(string))


# これで部位ごとに学習にかける．
print('\n\n****動作+レシピ****')

score = cross_val_score(clf, left_hip, left_hip_label, cv=5)
print('left_hip'+'\nCross-Validation scores: {}\n'.format(np.mean(score))+'-----------------------')
        
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
