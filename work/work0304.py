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



#5s
# 動作のみ

## データの読み込み ##
filename = 'left_hip.csv'
left_hip_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
left_hip_df = left_hip_df[left_hip_df['label'] == 'Take']
left_hip = left_hip_df.values[:, 2:8]
left_hip_label = left_hip_df.values[:, 1:2]


filename = 'right_arm.csv'
right_arm_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_arm_df = right_arm_df[right_arm_df['label'] == 'Take']
right_arm = right_arm_df.values[:, 2:8]
right_arm_label = right_arm_df.values[:, 1:2]

filename = 'left_wrist.csv'
left_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
left_wrist_df = left_wrist_df[left_wrist_df['label'] == 'Take']
left_wrist = left_wrist_df.values[:, 2:8]
left_wrist_label = left_wrist_df.values[:, 1:2]

filename = 'right_wrist.csv'
right_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_wrist_df = right_wrist_df[right_wrist_df['label'] == 'Take']
right_wrist = right_wrist_df.values[:, 2:8]
right_wrist_label = right_wrist_df.values[:, 1:2]




left_hip_train = left_hip
right_arm_train = right_arm
left_wrist_train = left_wrist
right_wrist_train = right_wrist


data = np.loadtxt("./train/left_hip/subject2_file_679.csv", delimiter=',', skiprows=1)
temp = []
left_hip_test = []
start = data[0][3]
for row in data:
    if row[3]-start > 5000:
        #書き込み
        left_hip_test.append(list(np.concatenate([np.mean(np.array(temp)[:, :3], axis=0), np.var(np.array(temp)[:, :3], axis=0)])))
        
        # スライドするので先頭を削除
        del temp[0]
        start = temp[0][3]
    temp.append(row[0:4])





"""
  
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