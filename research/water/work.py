import numpy as np
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.optim as optimizers
from model import Net
import matplotlib.pyplot as plt
import random
import os
os.chdir(os.path.dirname(__file__))


SOUND_DIR = './sounds/'
TRAIN_FILES = ['shampoo_1.mp3', 'shampoo_2.mp3']  # 学習用音源
TEST_FILE = 'shampoo_3.mp3'  # テスト用音源

EPOCH_NUM = 100  # 学習サイクル数
KERNEL_SIZE = 7  # カーネルサイズ（奇数のみ）
WINDOW_SIZE = 100000  # 1サンプルのサイズ
STEP = 1000  # 学習データのステップ幅
TEST_DATA_NUM = 100  # テストデータ数


def make_train_data():
    """
    学習データの作成
    """

    train_data, train_labels = [], []

    for filename in TRAIN_FILES:
        # 音源の読み出し
        sound = AudioSegment.from_file(SOUND_DIR + filename, 'mp3')

        # データの整形
        data = np.array(sound.get_array_of_samples())
        labels = np.linspace(0, 100, len(data))

        for index in range(0, len(data) - WINDOW_SIZE + 1, STEP):
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

    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DIR + TEST_FILE, 'mp3')

    # データの整形
    data = np.array(sound.get_array_of_samples())
    labels = np.linspace(0, 100, len(data))

    for index in range(TEST_DATA_NUM):
        start = random.randint(0, len(data) - WINDOW_SIZE)
        end = start + WINDOW_SIZE - 1
        test_data.append(data[start:end + 1])
        test_labels.append(labels[end])

    return test_data, test_labels


def main():
    def train():
        """
        モデルの学習
        """

        # 学習データの作成
        train_data, train_labels = make_train_data()
        # Tensorへ変換
        inputs = torch.tensor(
            train_data, dtype=torch.float, device=device).view(-1, 1, WINDOW_SIZE)
        labels = torch.tensor(
            train_labels, dtype=torch.float, device=device).view(-1, 1)

        model.train()
        print('\n***** 学習開始 *****')

        '''学習サイクル'''
        for epoch in range(EPOCH_NUM):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    'Epoch: {:d} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        model.eval()
        with torch.no_grad():
            # テストデータの作成
            test_data, test_labels = make_test_data()
            print(1)

        #     inputs, labels = data
        #     outputs = net(inputs)
        #     _, predicted = torch.max(outputs, 1)
        #     total += len(outputs)
        #     correct += (predicted == labels).sum().item()

        # print(
        #     f'correct: {correct}, accuracy: {correct} / {total} = {correct / total}')

    # モデルの構築
    model = Net(kernel_size=KERNEL_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)

    # モデルの学習
    train()

    # モデルのテスト
    test()


if __name__ == '__main__':
    # PyTorchの初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    main()
