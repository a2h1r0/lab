import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from model import Net
from preprocess import make_feature
from label_determination import majority_vote_sigmoid
import matplotlib.pyplot as plt
import csv
import glob
import re
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


DATA_DIR = './dataset/train/speed/1_13/'

TRAIN_SUBJECTS = ['1', '2']  # 学習に使用する被験者
TEST_SUBJECT = '3'  # テストに使用する被験者
USE_MARKERS = ['right_shoulder', 'left_wrist']

EPOCH_NUM = 1000000  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数
BATCH_SIZE = 500  # バッチサイズ
WINDOW_SIZE = 1000  # 1サンプルのサイズ
LABEL_THRESHOLD = 0.5


def make_train_data():
    """
    学習データの作成

    Returns:
        train_data (array): 学習データ
        train_labels (array): 学習データラベル
    """

    train_data, train_labels = [], []
    files = glob.glob(DATA_DIR + '/subject_[' + ''.join(TRAIN_SUBJECTS) + ']*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
        train_data.append(torch.tensor(feature_data, dtype=torch.float))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1]) - 1
        train_labels.append(multi_label_binarizer(label))

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成

    Returns:
        test_data (array): テストデータ
        test_labels (array): テストデータラベル
        answer_labels (array): テストデータ生ラベル
    """

    test_data, test_labels, answer_labels = [], [], []
    files = glob.glob(DATA_DIR + '/subject_' + TEST_SUBJECT + '*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
        test_data.append(torch.tensor(feature_data, dtype=torch.float))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1]) - 1
        test_labels.append(multi_label_binarizer(label))
        answer_labels.append(label)

    return test_data, test_labels, answer_labels


def get_marker_data(marker_index, data):
    """
    部位ごとのデータの取得

    Args:
        marker_index (int): 使用する部位のインデックス
        data (array): データ
    Returns:
        array: 部位ごとのデータ
    """

    return [row[marker_index] for row in data]


def multi_label_binarizer(label):
    """
    ラベルのワンホット化

    Args:
        label (int): ラベル
    Returns:
        array: ワンホットラベル
    """

    y = [0 for i in range(10)]
    y[label] = 1

    return y


def main():
    def train():
        """
        モデルの学習
        """

        for marker in range(len(USE_MARKERS)):
            loss_all = []

            # データの作成
            train_data = get_marker_data(marker, train_data_all)

            model.train()
            print('\n***** 学習開始 *****')

            for epoch in range(EPOCH_NUM):
                # Tensorへ変換
                inputs = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True).to(device)
                labels = torch.tensor(train_labels, dtype=torch.float, device=device)

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

        for marker in range(len(USE_MARKERS)):
            # データの作成
            test_data = get_marker_data(marker, test_data_all)

            model.eval()
            print('\n***** テスト *****')

            with torch.no_grad():
                # Tensorへ変換
                inputs = torch.nn.utils.rnn.pad_sequence(test_data, batch_first=True).to(device)
                labels = torch.tensor(test_labels, dtype=torch.float, device=device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 結果を整形
                prediction = outputs.to('cpu').detach().numpy().copy()
                predictions.append(prediction)

        # answer = labels.to('cpu').detach().numpy().copy()

        # # 予測と正解の差の合計を計算
        # diffs = np.abs(answer - prediction)
        # diff = np.sum(diffs) / len(diffs)

        # print('Diff: {:.3f} / Loss: {:.3f}\n'.format(diff, loss.item()))

    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = Net(input_size=21, hidden_size=HIDDEN_SIZE, out_features=10).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)

    # データの読み込み
    train_data_all, train_labels = make_train_data()
    test_data_all, test_labels, answer_labels = make_test_data()

    # モデルの学習
    train()

    # モデルのテスト
    predictions = []
    test()

    predictions = np.array(predictions).transpose(1, 0, 2)
    prediction_labels = []
    for prediction in predictions:
        prediction_labels.append(majority_vote_sigmoid(prediction, LABEL_THRESHOLD))

    # # Lossの描画
    # print('\nLossを描画します．．．')
    # plt.figure(figsize=(16, 9))
    # plt.plot(range(EPOCH_NUM), loss_all)
    # plt.xlabel('Epoch', fontsize=26)
    # plt.ylabel('Loss', fontsize=26)
    # plt.tick_params(labelsize=26)
    # # plt.savefig('./figures/loss.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    for answer, prediction in zip(prediction_labels, answer_labels):
        print('Answer: ' + str(answer) + ' / Prediction: ' + str(prediction))


if __name__ == '__main__':
    main()
