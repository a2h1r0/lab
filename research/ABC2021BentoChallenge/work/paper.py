import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import NetAll
from preprocess import make_feature
from label_determination import majority_vote_sigmoid
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import glob
import re
import time
import random
import copy
import sys
import os
os.chdir(os.path.dirname(__file__))


FILENAME = '../dataset/train/autocorrelation/'
TEST_DATA_DIR = '../dataset/test/autocorrelation/'

USE_MARKERS = ['right_shoulder', 'right_elbow', 'right_wrist',
               'left_shoulder', 'left_elbow', 'left_wrist']

NUM_CLASSES = 10  # クラス数
EPOCH_NUM = 5000  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数
LABEL_THRESHOLD = 0.0  # ラベルを有効にする閾値


def make_train_data():
    """
    学習データの作成

    Returns:
        array: 学習データ
        array: 学習データラベル
        array: 学習データファイル
    """

    train_data, train_labels = [], []
    files = glob.glob(TRAIN_DATA_DIR + '/*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
            if len(feature_data[0]) < 5:
                continue
        train_data.append(torch.tensor(feature_data, dtype=torch.float, device=device))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
        train_labels.append(multi_label_binarizer(label))

    return train_data, train_labels


def main():

    # Lossの描画
    figures_dir = '../figures/'
    plt.figure(figsize=(16, 9))
    for marker, loss in zip(USE_MARKERS, loss_all):
        plt.plot(range(1, EPOCH_NUM + 1), loss, label=marker)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.legend(fontsize=26, loc='upper right')
    plt.tick_params(labelsize=26)
    plt.savefig(figures_dir + 'prediction_loss.svg', bbox_inches='tight', pad_inches=0)
    plt.savefig(figures_dir + 'prediction_loss.eps', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    main()
