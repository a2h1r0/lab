import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


EPOCH = 10  # エポック数

FEATURE_SIZE = 4  # 特徴量次元数
NUM_CLASSES = 2  # 分類クラス数

EPOCH_NUM = 100  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数


def make_train_data():
    """
    学習データの作成

    Returns:
        array: 学習データ
        array: 学習データラベル
    """

    train_data, train_labels = [], []
    files = glob.glob(
        '{0}/drink/subject_[{1}]/*/*.csv'.format(DATA_DIR, ''.join(TRAIN_SUBJECTS)))
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            data = [list(map(lambda value: float(value), row[3:8]))
                    for row in reader]
        train_data.append(torch.tensor(
            data[:-1], dtype=torch.float, device=device))
        train_labels.append(get_label(filename))

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成

    Returns:
        array: テストデータ
        array: テストデータラベル
    """

    test_data, test_labels = [], []
    files = glob.glob(
        '{0}/drink/subject_[{1}]/*/*.csv'.format(DATA_DIR, ''.join(TEST_SUBJECT)))
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            data = [list(map(lambda value: float(value), row[3:8]))
                    for row in reader]
        test_data.append(torch.tensor(
            data[:-1], dtype=torch.float, device=device))
        test_labels.append(get_label(filename))

    return test_data, test_labels


def get_label(filename):
    """
    ラベルの作成

    Args:
        filename (string): 読み込んだファイル名
    Returns:
        int: ラベル
    """

    return int('drunk' in filename)


def sigmoid_to_label(prediction):
    """
    シグモイド予測値のラベル化

    Args:
        label (int): シグモイド予測
    Returns:
        array: 結果ラベル
    """

    # 0.5とかで切ったら良さそう
    return np.argmax(prediction) + 1


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

        for epoch in range(EPOCH):
            inputs = torch.tensor(
                train_data, dtype=torch.float, device=device).view(-1, 1, FEATURE_SIZE)
            labels = torch.tensor(
                train_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        # データの読み込み
        test_data, test_labels = make_test_data()

        model.eval()
        print('\n***** テスト *****')

        with torch.no_grad():
            inputs = torch.tensor(
                test_data, dtype=torch.float, device=device).view(-1, 1, FEATURE_SIZE)
            labels = torch.tensor(test_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = sigmoid(outputs)

            labels = labels.to('cpu').detach().numpy().copy()
            outputs = outputs.to('cpu').detach().numpy().copy()
            answers, predictions = [], []
            for label, output in zip(labels, outputs):
                answers.append(label)
                predictions.append(sigmoid_to_label(output))

    train()
    test()


if __name__ == '__main__':
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = './result/{}.csv'.format(now)
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        loss_all, predictions, answers = [], [], []

        main()

        # todo: データ作成まで完成．学習開始処理から確認する．
        # このへん考える
        # 一旦loss_all, predictions, answersが正しいか確認し，データ保存処理考える
        # result_writer.writerow(['(Avg.)' + BOTTLE, sum(scores) / len(scores)])
