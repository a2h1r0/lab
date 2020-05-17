import numpy as np
import pandas as pd
import matplotlib
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
import csv
from sklearn import svm
from sklearn.model_selection import cross_val_score
warnings.simplefilter('ignore')



clf = OneClassSVM(kernel="rbf", gamma='auto')


#5s
# 動作のみ

## データの読み込み ##
filename = './data_5s/right_wrist.csv'
right_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_wrist_df = right_wrist_df[right_wrist_df['label'] == 'Take']
right_wrist = right_wrist_df.values[:, 2:8]
right_wrist_label = right_wrist_df.values[:, 1:2]

right_wrist_train = right_wrist


#wash_take = ['subject2_file_679', 'subject2_file_52', 'subject2_file_485', 'subject2_file_263', 'subject2_file_792', 'subject2_file_635', 'subject3_file_537', 'subject3_file_901', 'subject3_file_389', 'subject3_file_829', 'subject1_file_780', 'subject1_file_319']


with warnings.catch_warnings():
    warnings.filterwarnings("error")
        
    print('\n---right_wrist---\n')
    try:
        data = np.loadtxt('./train/right_wrist/'+'subject2_file_52'+'.csv', delimiter=',', skiprows=1)
        temp = []
        right_wrist_test = []
        start = data[0][3]
        for row in data:
            if row[3]-start > 5000:
                #書き込み
                right_wrist_test.append(list(np.concatenate([np.mean(np.array(temp)[:, :3], axis=0), np.var(np.array(temp)[:, :3], axis=0)])))
                right_wrist_test[-1].extend([start, temp[-1][3]])
                
                # スライドするので先頭を削除
                while row[3]-temp[0][3] > 5000:
                    del temp[0]
                    if temp == []:
                        break
                if temp == []:
                    start = row[3]
                else:
                    start = temp[0][3]
            temp.append(row[0:4])
    except IOError:
        sys.exit()
    except UserWarning: 
        sys.exit()
  
    # OneClassSVM
    clf.fit(right_wrist_train)
    pred = clf.predict(np.array(right_wrist_test)[:, :6])
    plt.figure()
    plt.plot(range(len(pred)), list(pred))
    print('OK\n------------')


with open('subject2_file_52'+'_out.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    
    writer.writerow(['mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z', 'start', 'finish'])
    # startは範囲開始，fnishは計算範囲の最後

    for index in range(len(pred)-1):
        if pred[index] == 1:
            writer.writerow(right_wrist_test[index])
            # 変化時に改行，インデックスエラー回避のためrangeは-1
            if pred[index+1] == -1:
                writer.writerow('\n')
    writer.writerow(right_wrist_test[index+1])