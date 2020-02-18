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



file_num = ['3', '11', '21']
directory = ['left_hip', 'right_arm', 'left_wrist', 'right_wrist']

take_set = [[] for i in range(len(file_num))]
for part in directory:
    for index, num in enumerate(file_num):
        path = glob.glob('.\\'+part+'\\subject?_file_'+num+'.csv')
        for filename in path:
            data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=[0,1,2])
            if data == []:
                continue
            take_set[index] = np.append(take_set, np.mean(data, axis=0))
            take_set[index] = np.append(take_set, np.var(data, axis=0))

