import numpy as np
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.optim as optimizers
from model import Net
import matplotlib.pyplot as plt
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


SAMPLING_RATE = 48000
SOUND_DIR = './sounds/trimmed/' + str(SAMPLING_RATE) + '/'

COFFEE = ['coffee_1.mp3', 'coffee_2.mp3', 'coffee_3.mp3',
          'coffee_4.mp3', 'coffee_5.mp3', 'coffee_6.mp3']
DETERGENT = ['detergent_1.mp3', 'detergent_2.mp3', 'detergent_3.mp3',
             'detergent_4.mp3', 'detergent_5.mp3', 'detergent_6.mp3']
SHAMPOO = ['shampoo_1.mp3', 'shampoo_2.mp3', 'shampoo_3.mp3',
           'shampoo_4.mp3', 'shampoo_5.mp3', 'shampoo_6.mp3']
SKINMILK = ['skinmilk_1.mp3', 'skinmilk_2.mp3', 'skinmilk_3.mp3',
            'skinmilk_4.mp3', 'skinmilk_5.mp3', 'skinmilk_6.mp3']
TOKKURI = ['tokkuri_1.mp3', 'tokkuri_2.mp3', 'tokkuri_3.mp3',
           'tokkuri_4.mp3', 'tokkuri_5.mp3', 'tokkuri_6.mp3']

TEST_FILE_NUM = 1  # テストに使うファイル数

TRAIN_FILES = COFFEE[:-TEST_FILE_NUM]  # 学習用音源
TEST_FILES = COFFEE[-TEST_FILE_NUM:]  # テスト用音源
# TRAIN_FILES = COFFEE[:-TEST_FILE_NUM] + DETERGENT[:-TEST_FILE_NUM] + \
#     SHAMPOO[:-TEST_FILE_NUM] + SKINMILK[:-TEST_FILE_NUM] + \
#     TOKKURI[:-TEST_FILE_NUM]  # 学習用音源
# TEST_FILES = COFFEE[-TEST_FILE_NUM:] + DETERGENT[-TEST_FILE_NUM:] + \
#     SHAMPOO[-TEST_FILE_NUM:] + SKINMILK[-TEST_FILE_NUM:] + \
#     TOKKURI[-TEST_FILE_NUM:]  # テスト用音源

EPOCH_NUM = 500  # 学習サイクル数
KERNEL_SIZE = 5  # カーネルサイズ（奇数のみ）
BATCH_SIZE = 200  # バッチサイズ
WINDOW_SECOND = 1.0  # 1サンプルの秒数
WINDOW_SIZE = int(WINDOW_SECOND * SAMPLING_RATE)  # 1サンプルのサイズ
TEST_ONEFILE_DATA_NUM = 100  # 1ファイルごとのテストデータ数


def check_sampling_rate():
    """
    サンプリング周波数の確認
    """

    for filename in TRAIN_FILES + TEST_FILES:
        # 音源の読み出し
        sound = AudioSegment.from_file(SOUND_DIR + filename, 'mp3')
        if len(sound[:1000].get_array_of_samples()) != SAMPLING_RATE:
            print('\n' + filename + 'のサンプリングレートが異なります．\n')
            sys.exit()


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
        if not index in history:
            history.append(index)
            random_data.append(data[index])
            random_labels.append(labels[index])

    return random_data, random_labels


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
            random_data, random_labels = get_random_data(
                'train', train_data, train_labels)
            # Tensorへ変換
            inputs = torch.tensor(
                random_data, dtype=torch.float, device=device).view(-1, 1, WINDOW_SIZE)
            labels = torch.tensor(
                random_labels, dtype=torch.float, device=device).view(-1, 1)

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
            random_data, random_labels = get_random_data(
                'test', test_data, test_labels)
            # Tensorへ変換
            inputs = torch.tensor(
                random_data, dtype=torch.float, device=device).view(-1, 1, WINDOW_SIZE)
            labels = torch.tensor(
                random_labels, dtype=torch.float, device=device).view(-1, 1)

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
    main()
