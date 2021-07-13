import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from model import Net
from preprocess import make_feature
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

EPOCH_NUM = 1000  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数
BATCH_SIZE = 500  # バッチサイズ
WINDOW_SIZE = 1000  # 1サンプルのサイズ


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
        train_data.append(feature_data)
        activity = re.findall(r'activity_\d+', filename)[0]
        label = activity.split('_')[1]
        train_labels.append(label)

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成

    Returns:
        test_data (array): テストデータ
        test_labels (array): テストデータラベル
    """

    test_data, test_labels = [], []
    files = glob.glob(DATA_DIR + '/subject_' + TEST_SUBJECT + '*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
        test_data.append(feature_data)
        activity = re.findall(r'activity_\d+', filename)[0]
        label = activity.split('_')[1]
        test_labels.append(label)

    return test_data, test_labels


def get_random_data(mode, data, labels):
    """
    ランダムデータの取得

    Args:
        mode (string): train or test
        data (array): データ
        labels (array): ラベル
    Returns:
        array: ランダムデータ
        array: ラベル
    """

    if mode == 'train':
        data_size = BATCH_SIZE
    elif mode == 'test':
        data_size = TEST_ONEFILE_DATA_NUM

    history = []
    random_data, random_labels = [], []
    while len(random_data) < data_size:
        index = random.randint(0, len(data) - 1)
        if index not in history:
            history.append(index)
            if FFT == True:
                random_data.append(fft(data[index]))
            else:
                random_data.append(data[index])
            random_labels.append(labels[index])

    return random_data, random_labels


def main():
    def train():
        """
        モデルの学習
        """

        # データの読み込み
        train_data, train_labels = make_train_data()

        model.train()
        print('\n***** 学習開始 *****')

        for epoch in range(EPOCH_NUM):
            # 学習データの作成
            random_data, random_labels = get_random_data('train', train_data, train_labels)
            # Tensorへ変換
            inputs = torch.tensor(random_data, dtype=torch.float, device=device).view(-1, 1, WINDOW_SIZE)
            labels = torch.tensor(random_labels, dtype=torch.float, device=device).view(-1, 1)

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
            # テストデータの作成
            random_data, random_labels = get_random_data('test', test_data, test_labels)
            # Tensorへ変換
            inputs = torch.tensor(random_data, dtype=torch.float, device=device).view(-1, 1, WINDOW_SIZE)
            labels = torch.tensor(random_labels, dtype=torch.float, device=device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 結果を整形
            prediction = outputs.to('cpu').detach().numpy().copy()
            prediction = prediction.reshape(-1)
            answer = labels.to('cpu').detach().numpy().copy()
            answer = answer.reshape(-1)

            # 予測と正解の差の合計を計算
            diffs = np.abs(answer - prediction)
            diff = np.sum(diffs) / len(diffs)

            print('Diff: {:.3f} / Loss: {:.3f}\n'.format(diff, loss.item()))

    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = Net(input_size=21, hidden_size=HIDDEN_SIZE, out_features=10).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)

    # モデルの学習
    loss_all = []
    train()

    # モデルのテスト
    test()

    # Lossの描画
    print('\nLossを描画します．．．')
    plt.figure(figsize=(16, 9))
    plt.plot(range(EPOCH_NUM), loss_all)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.tick_params(labelsize=26)
    # plt.savefig('./figures/loss.png', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
