import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import Net
from preprocess import make_feature
from label_determination import majority_vote_sigmoid
import matplotlib.pyplot as plt
import csv
import glob
import re
import datetime
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


DATA_DIR = '../dataset/train/speed/1_13/'

TRAIN_SUBJECTS = ['1', '2']  # 学習に使用する被験者
TEST_SUBJECT = '3'  # テストに使用する被験者
USE_MARKERS = ['right_shoulder', 'left_wrist']

EPOCH_NUM = 100  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数
BATCH_SIZE = 500  # バッチサイズ
WINDOW_SIZE = 1000  # 1サンプルのサイズ
LABEL_THRESHOLD = 0.1


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
        train_data.append(torch.tensor(feature_data, dtype=torch.float, device=device))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
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
        test_data.append(torch.tensor(feature_data, dtype=torch.float, device=device))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
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
    y[label - 1] = 1

    return y


def main():
    def train():
        """
        モデルの学習
        """

        # データの作成
        train_data = get_marker_data(marker, train_data_all)
        train_data_length = [len(data) for data in train_data]

        model.train()
        print('\n***** 学習開始 *****')

        for epoch in range(EPOCH_NUM):
            # パディング処理
            padded_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True)
            inputs = torch.nn.utils.rnn.pack_padded_sequence(padded_data, train_data_length, batch_first=True, enforce_sorted=False)
            labels = torch.tensor(train_labels, dtype=torch.float, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_all[-1].append(loss.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        # データの作成
        test_data = get_marker_data(marker, test_data_all)

        model.eval()
        print('\n***** テスト *****')

        with torch.no_grad():
            for input in test_data:
                output = model(input.view(1, len(input), -1))
                predictions[-1].append(output.to('cpu').detach().numpy().copy().squeeze())

        # answer = labels.to('cpu').detach().numpy().copy()

        # # 予測と正解の差の合計を計算
        # diffs = np.abs(answer - prediction)
        # diff = np.sum(diffs) / len(diffs)

        # print('Diff: {:.3f} / Loss: {:.3f}\n'.format(diff, loss.item()))

    def label_determination(predictions):
        """
        ラベルのワンホット化

        Args:
            predictions (array): 部位ごとの予測
        Returns:
            array: 予測結果
        """

        predictions = np.array(predictions).transpose(1, 0, 2)
        labels = []
        for prediction in predictions:
            labels.append(majority_vote_sigmoid(prediction, LABEL_THRESHOLD))

        return labels

    # モデルの構築
    model = Net(input_size=21, hidden_size=HIDDEN_SIZE, out_features=10).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)

    # データの読み込み
    train_data_all, train_labels = make_train_data()
    test_data_all, test_labels, answer_labels = make_test_data()

    loss_all = []
    predictions = []
    for marker in range(len(USE_MARKERS)):
        print('\n!!!!! ' + USE_MARKERS[marker] + ' !!!!!')

        # モデルの学習
        loss_all.append([])
        train()

        # モデルのテスト
        predictions.append([])
        test()

    # 予測ラベルの決定
    prediction_labels = label_determination(predictions)

    # 結果の表示
    for answer, prediction in zip(answer_labels, prediction_labels):
        print('Answer: ' + str(answer - 1) + ' / Prediction: ' + str(prediction - 1))
    print(classification_report(answer_labels, prediction_labels))

    # Lossの保存
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    loss_file = '../data/loss_' + now + '.csv'
    with open(loss_file, 'w', newline='') as f:
        loss_writer = csv.writer(f)
        loss_writer.writerow(['Epoch'] + USE_MARKERS)

        for epoch, loss in enumerate(np.array(loss_all).T):
            loss_writer.writerow([epoch + 1] + list(loss))

    # 結果の描画
    print('\n結果を描画します．．．')
    plt.figure()
    sns.heatmap(confusion_matrix(answer_labels, prediction_labels))
    plt.savefig('../figures/result_' + now + '.png', bbox_inches='tight', pad_inches=0)

    # Lossの描画
    plt.figure(figsize=(16, 9))
    for marker, loss in zip(USE_MARKERS, loss_all):
        plt.plot(range(1, EPOCH_NUM + 1), loss, label=marker)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.legend(fontsize=26, loc='upper right')
    plt.tick_params(labelsize=26)
    plt.savefig('../figures/loss_' + now + '.png', bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    main()
