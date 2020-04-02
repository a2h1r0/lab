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

import glob


labels_df = pd.read_csv("labels.txt", usecols=[0, 1, 2, 3, 4, 5, 6, 7], encoding='Shift-JIS').fillna(0)


if __name__ == '__main__':
    file_list = glob.glob('./left_hip/subject1*.csv')

    sub1 = []
    lens_left_hip_sub1 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub1.append(len(df))
        for _, row in labels_df.iterrows():
            if row[0] == filename.strip('./left_hip\\').rstrip('.csv'):
                lens_left_hip_sub1.append(len([i for i in row if i != 0])-2)
                
                
if __name__ == '__main__':
    file_list = glob.glob('./left_hip/subject2*.csv')

    sub2 = []
    files_left_hip_sub2 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub2.append(len(df))
        files_left_hip_sub2.append(filename.strip('./left_hip\\'))

if __name__ == '__main__':
    file_list = glob.glob('./left_hip/subject3*.csv')

    sub3 = []
    files_left_hip_sub3 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub3.append(len(df))
        files_left_hip_sub3.append(filename.strip('./left_hip\\'))

print("---***left_hip***---\n")
print("--subject1--")
print("最大：", end="")
print(np.max(np.array(sub1)))
print("平均：", end="")
print(np.mean(np.array(sub1)))
print("最小：", end="")
print(np.min(np.array(sub1)))
print("\n")

print("--subject2--")
print("最大：", end="")
print(np.max(np.array(sub2)))
print("平均：", end="")
print(np.mean(np.array(sub2)))
print("最小：", end="")
print(np.min(np.array(sub2)))
print("\n")

print("--subject3--")
print("最大：", end="")
print(np.max(np.array(sub3)))
print("平均：", end="")
print(np.mean(np.array(sub3)))
print("最小：", end="")
print(np.min(np.array(sub3)))
print("\n\n")

"""
if __name__ == '__main__':
    file_list = glob.glob('./left_wrist/subject1*.csv')

    sub1 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub1.append(len(df))

if __name__ == '__main__':
    file_list = glob.glob('./left_wrist/subject2*.csv')

    sub2 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub2.append(len(df))

if __name__ == '__main__':
    file_list = glob.glob('./left_wrist/subject3*.csv')

    sub3 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub3.append(len(df))

print("---***left_wrist***---\n")
print("--subject1--")
print("最大：", end="")
print(np.max(np.array(sub1)))
print("平均：", end="")
print(np.mean(np.array(sub1)))
print("最小：", end="")
print(np.min(np.array(sub1)))
print("\n")

print("--subject2--")
print("最大：", end="")
print(np.max(np.array(sub2)))
print("平均：", end="")
print(np.mean(np.array(sub2)))
print("最小：", end="")
print(np.min(np.array(sub2)))
print("\n")

print("--subject3--")
print("最大：", end="")
print(np.max(np.array(sub3)))
print("平均：", end="")
print(np.mean(np.array(sub3)))
print("最小：", end="")
print(np.min(np.array(sub3)))
print("\n\n")


if __name__ == '__main__':
    file_list = glob.glob('./right_wrist/subject1*.csv')

    sub1 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub1.append(len(df))

if __name__ == '__main__':
    file_list = glob.glob('./right_wrist/subject2*.csv')

    sub2 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub2.append(len(df))

if __name__ == '__main__':
    file_list = glob.glob('./right_wrist/subject3*.csv')

    sub3 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub3.append(len(df))

print("---***right_wrist***---\n")
print("--subject1--")
print("最大：", end="")
print(np.max(np.array(sub1)))
print("平均：", end="")
print(np.mean(np.array(sub1)))
print("最小：", end="")
print(np.min(np.array(sub1)))
print("\n")

print("--subject2--")
print("最大：", end="")
print(np.max(np.array(sub2)))
print("平均：", end="")
print(np.mean(np.array(sub2)))
print("最小：", end="")
print(np.min(np.array(sub2)))
print("\n")

print("--subject3--")
print("最大：", end="")
print(np.max(np.array(sub3)))
print("平均：", end="")
print(np.mean(np.array(sub3)))
print("最小：", end="")
print(np.min(np.array(sub3)))
print("\n\n")


if __name__ == '__main__':
    file_list = glob.glob('./right_arm/subject1*.csv')

    sub1 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub1.append(len(df))

if __name__ == '__main__':
    file_list = glob.glob('./right_arm/subject2*.csv')

    sub2 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub2.append(len(df))

if __name__ == '__main__':
    file_list = glob.glob('./right_arm/subject3*.csv')

    sub3 = []
    for filename in file_list:
        df = pd.read_csv(filename, usecols=['X'], encoding='Shift-JIS')
        sub3.append(len(df))

print("---***right_arm***---\n")
print("--subject1--")
print("最大：", end="")
print(np.max(np.array(sub1)))
print("平均：", end="")
print(np.mean(np.array(sub1)))
print("最小：", end="")
print(np.min(np.array(sub1)))
print("\n")

print("--subject2--")
print("最大：", end="")
print(np.max(np.array(sub2)))
print("平均：", end="")
print(np.mean(np.array(sub2)))
print("最小：", end="")
print(np.min(np.array(sub2)))
print("\n")

print("--subject3--")
print("最大：", end="")
print(np.max(np.array(sub3)))
print("平均：", end="")
print(np.mean(np.array(sub3)))
print("最小：", end="")
print(np.min(np.array(sub3)))
print("\n\n")
"""
