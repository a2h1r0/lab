import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from model import Net
import matplotlib.pyplot as plt
import csv
import glob
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


DATA_DIR = './Train_data/'

TRAIN_SUBJECTS = ['1', '2']  # 学習に使用する被験者
TEST_SUBJECT = 'subject_3'  # テストに使用する被験者

EPOCH_NUM = 1000  # 学習サイクル数
HIDDEN_SIZE = 5  # 隠れ層数
BATCH_SIZE = 500  # バッチサイズ
WINDOW_SIZE = 1000  # 1サンプルのサイズ


def read_data():
    """
    データの読み込み

    Returns:
        train_data (array): 学習データ
        test_data (array): テストデータ
    """

    train_data = []
    files = glob.glob(DATA_DIR + '/subject_[' + ''.join(TRAIN_SUBJECTS) + ']*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            # ここで特徴量を選択
            train_data = [row for row in reader]

    test_data = []
    filename = glob.glob(DATA_DIR + '/subject_' + TEST_SUBJECT + '*.csv')
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        # ここで特徴量を選択
        test_data = [row for row in reader]

    return train_data, test_data


def make_train_data():
    """
    学習データの作成
    """

    train_data, train_labels = [], []

    for filename in TRAIN_FILES:
        # 音源の読み出し
        sound = AudioSegment.from_file(SOUND_DIR + filename, 'mp3')
        data = np.array(sound.get_array_of_samples())
        labels = np.linspace(0, 100, len(data))

        for index in range(0, len(data) - WINDOW_SIZE + 1):
            start = index
            end = start + WINDOW_SIZE - 1
            train_data.append(data[start:end + 1])
            train_labels.append(labels[end])

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成
    """

    test_data, test_labels = [], []

    for filename in TEST_FILES:
        # 音源の読み出し
        sound = AudioSegment.from_file(SOUND_DIR + filename, 'mp3')
        data = np.array(sound.get_array_of_samples())
        labels = np.linspace(0, 100, len(data))

        for index in range(0, len(data) - WINDOW_SIZE + 1):
            start = index
            end = start + WINDOW_SIZE - 1
            test_data.append(data[start:end + 1])
            test_labels.append(labels[end])

    return test_data, test_labels


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
            predict = outputs.to('cpu').detach().numpy().copy()
            predict = predict.reshape(-1)
            answer = labels.to('cpu').detach().numpy().copy()
            answer = answer.reshape(-1)

            # 予測と正解の差の合計を計算
            diffs = np.abs(answer - predict)
            diff = np.sum(diffs) / len(diffs)

            print('Diff: {:.3f} / Loss: {:.3f}\n'.format(diff, loss.item()))

    # ファイルの検証
    check_sampling_rate()

    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = Net(kernel_size=KERNEL_SIZE).to(device)
    criterion = nn.MSELoss()
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
    # plt.savefig('./figure/loss.png', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    read_data()
    # main()
