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
filename = './data_10s/right_wrist.csv'
right_wrist_df = pd.read_csv(filename, usecols=['recipe', 'label', 'mean_X', 'mean_Y', 'mean_Z', 'var_X', 'var_Y', 'var_Z'], encoding='Shift-JIS').dropna()
right_wrist_df = right_wrist_df[right_wrist_df['label'] == 'Take']
right_wrist = right_wrist_df.values[:, 2:8]
right_wrist_label = right_wrist_df.values[:, 1:2]

right_wrist_train = right_wrist


take = ['subject2_file_457', 'subject2_file_646', 'subject2_file_743', 'subject2_file_308', 'subject2_file_325', 'subject2_file_540', 'subject2_file_138', 'subject2_file_452', 'subject2_file_923', 'subject2_file_49', 'subject2_file_588', 'subject2_file_171', 'subject2_file_522', 'subject2_file_775', 'subject2_file_175', 'subject2_file_917', 'subject2_file_328', 'subject2_file_642', 'subject3_file_867', 'subject3_file_850', 'subject3_file_97', 'subject3_file_554', 'subject3_file_34', 'subject3_file_264', 'subject3_file_264', 'subject3_file_808', 'subject3_file_216', 'subject3_file_382', 'subject3_file_817', 'subject3_file_5', 'subject3_file_418', 'subject3_file_9', 'subject3_file_531', 'subject3_file_318', 'subject3_file_849', 'subject3_file_749', 'subject3_file_511', 'subject3_file_853', 'subject3_file_557', 'subject3_file_977', 'subject1_file_622', 'subject1_file_869', 'subject1_file_588', 'subject1_file_140', 'subject1_file_808', 'subject1_file_238', 'subject1_file_618', 'subject1_file_964', 'subject1_file_884', 'subject1_file_243', 'subject1_file_957', 'subject1_file_873', 'subject1_file_37', 'subject1_file_417', 'subject1_file_979', 'subject1_file_348', 'subject1_file_162', 'subject1_file_969', 'subject1_file_320', 'subject1_file_385', 'subject1_file_976', 'subject1_file_309', 'subject1_file_211']

#wash_take = ['subject2_file_679', 'subject2_file_52', 'subject2_file_485', 'subject2_file_263', 'subject2_file_792', 'subject2_file_635', 'subject3_file_537', 'subject3_file_901', 'subject3_file_389', 'subject3_file_829', 'subject1_file_780', 'subject1_file_319']


with warnings.catch_warnings():
    warnings.filterwarnings("error")
        
    print('\n---right_wrist---\n')
    for file in take:
        try:
            data = np.loadtxt('./train/right_wrist/'+file+'.csv', delimiter=',', skiprows=1)
            temp = []
            right_wrist_test = []
            start = data[0][3]
            for row in data:
                if row[3]-start > 5000:
                    #書き込み
                    right_wrist_test.append(list(np.concatenate([np.mean(np.array(temp)[:, :3], axis=0), np.var(np.array(temp)[:, :3], axis=0)])))
                    
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
            continue
        except UserWarning: 
            continue
      
        # OneClassSVM
        clf.fit(right_wrist_train)
        pred = clf.predict(right_wrist_test)
        plt.figure()
        plt.plot(range(len(pred)), list(pred))
        plt.savefig('./figures/right_wrist_own/'+file+'.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    print('OK\n------------')
