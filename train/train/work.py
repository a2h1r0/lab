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


file_num = ['3', '11', '21']
directory = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']


take_set = [[[] for i in range(len(file_num))] for i in range(len(directory))]
for part, part_name in enumerate(directory):
    for file_index, num in enumerate(file_num):
        path = glob.glob('.\\'+part_name+'\\subject?_file_'+num+'.csv')
        for filename in path:
            try:data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2]) 
            except IOError:continue
            except UserWarning:continue
            take_set[part][file_index].append([0 for i in range(6)])
            take_set[part][file_index][-1] = list(np.concatenate([np.mean(data, axis=0), np.var(data, axis=0)]))


