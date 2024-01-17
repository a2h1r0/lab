import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optimizers
from model import Net
import matplotlib.pyplot as plt
import math
import csv
import glob
import re
import datetime
import random
import copy
import sys
import os
os.chdir(os.path.dirname(__file__))


DATA_DIR = './data/'
TRAIN_SUBJECTS = ['1', '2']
TEST_SUBJECT = '3'


EPOCH = 2  # エポック数

FEATURE_SIZE = 6  # 特徴量次元数
NUM_CLASSES = 1  # 分類クラス数

EPOCH_NUM = 100  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数


def make_train_data():
    """
    学習データの作成

    Returns:
        list: 学習データ
        list: 学習データラベル
    """

    train_data, train_labels = [], []
    files = glob.glob(
        '{0}/drink/subject_[{1}]/*/*.csv'.format(DATA_DIR, ''.join(TRAIN_SUBJECTS)))
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            data = [list(map(lambda value: float(value), row[2:8]))
                    for row in reader]
        train_data.append(torch.tensor(
            data[:-1], dtype=torch.float, device=device))
        train_labels.append(label_to_onehot(filename))

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成

    Returns:
        list: テストデータ
        list: テストデータラベル
        list: テストファイル名
    """

    test_data, test_labels = [], []
    files = glob.glob(
        '{0}/drink/subject_[{1}]/*/*.csv'.format(DATA_DIR, ''.join(TEST_SUBJECT)))
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            data = [list(map(lambda value: float(value), row[2:8]))
                    for row in reader]
        test_data.append(torch.tensor(
            data[:-1], dtype=torch.float, device=device))
        test_labels.append(label_to_onehot(filename))

    return test_data, test_labels, files


def label_to_onehot(label):
    """
    ワンホットラベルの作成

    Args:
        label (string): 読み込んだラベル
    Returns:
        list: ワンホットラベル
    """

    if NUM_CLASSES == 1:
        label = [int('drunk' in label)]

    return label


def sigmoid_to_onehot(prediction):
    """
    シグモイド予測値のワンホットラベル化

    Args:
        prediction (list): シグモイド予測
    Returns:
        list: ワンホットラベル
    """

    return list(map(lambda value: int(value > 0.5), prediction))


def onehot_to_label(classes):
    """
    ワンホットラベルのラベル化

    Args:
        classes (list): ワンホットラベル
    Returns:
        int: ラベル
    """

    if NUM_CLASSES == 1:
        label = int(classes[0])

    return label


def main():
    # モデルの構築
    model = Net(input_size=FEATURE_SIZE, output_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(model.parameters())
    sigmoid = nn.Sigmoid()

    def train():
        """
        モデルの学習
        """

        # データの読み込み
        train_data, train_labels = make_train_data()

        model.train()
        print('\n***** 学習開始 *****')

        loss_all = []
        for epoch in range(EPOCH):
            inputs = torch.tensor(rnn.pad_sequence(
                train_data), dtype=torch.float, device=device).view(len(train_data), FEATURE_SIZE, -1)
            labels = torch.tensor(
                train_labels, dtype=torch.float, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

        return loss_all

    def test():
        """
        モデルのテスト
        """

        # データの読み込み
        test_data, test_labels, test_files = make_test_data()

        model.eval()
        print('\n***** テスト *****')

        predictions, answers = [], []
        with torch.no_grad():
            inputs = torch.tensor(rnn.pad_sequence(
                test_data), dtype=torch.float, device=device).view(len(test_data), FEATURE_SIZE, -1)
            labels = torch.tensor(
                test_labels, dtype=torch.float, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = sigmoid(outputs)

            labels = labels.to('cpu').detach().numpy().copy()
            outputs = outputs.to('cpu').detach().numpy().copy()
            for label, output in zip(labels, outputs):
                answers.append(onehot_to_label(label))
                predictions.append(onehot_to_label(sigmoid_to_onehot(output)))

        return predictions, answers, test_files

    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = './result/{}.csv'.format(now)
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        loss_all = train()
        predictions, answers, test_files = test()

        for filename, answer, prediction in zip(test_files, answers, predictions):
            result_writer.writerow(
                [filename.split('\\')[-1], answer, prediction])


if __name__ == '__main__':
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    main()
