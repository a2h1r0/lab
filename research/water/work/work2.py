import numpy as np
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.optim as optimizers
import model as models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from natsort import natsorted
import scipy
import librosa
import glob
import csv
import datetime
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


BOTTLE = 'shampoo2'

SOUND_DIR = '../sounds/raw/' + BOTTLE + '/'


EPOCH_NUM = 500  # 学習サイクル数
KERNEL_SIZE = 3  # カーネルサイズ（奇数のみ）
BATCH_SIZE = 10000  # バッチサイズ
WINDOW_SECOND = 0.2  # 1サンプルの秒数
STEP = 10000  # スライド幅
TEST_ONEFILE_DATA_NUM = 1000  # 1ファイルごとのテストデータ数


def get_sampling_rate():
    """
    サンプリング周波数の取得
    """

    sound = AudioSegment.from_file(SOUND_DIR + TRAIN_FILES[0], 'mp3')

    return len(sound[:1000].get_array_of_samples())


def mfcc(sound_data):
    """
    MFCC

    Args:
        sound_data (:obj:`ndarray`): 音データ
    Returns:
        array: MFCC特徴量配列
    """

    mfccs = librosa.feature.mfcc(sound_data, sr=SAMPLING_RATE, n_fft=512)
    mfccs = np.average(mfccs, axis=1)

    return mfccs


def make_train_data():
    """
    学習データの作成
    """

    train_data, train_labels = [], []

    for filename in TRAIN_FILES:
        # 音源の読み出し
        sound, _ = librosa.load(SOUND_DIR + filename, sr=SAMPLING_RATE)
        amounts = np.linspace(0, 100, len(sound))

        for index in range(0, len(sound) - WINDOW_SIZE + 1, STEP):
            start = index
            end = start + WINDOW_SIZE - 1
            train_data.append(mfcc(sound[start:end + 1]))
            train_labels.append(label_binarizer(amounts[end]))

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成
    """

    test_data, test_labels = [], []

    # 音源の読み出し
    sound, _ = librosa.load(SOUND_DIR + TEST_FILE, sr=SAMPLING_RATE)
    amounts = np.linspace(0, 100, len(sound))

    for index in range(0, len(sound) - WINDOW_SIZE + 1, STEP):
        start = index
        end = start + WINDOW_SIZE - 1
        test_data.append(mfcc(sound[start:end + 1]))
        test_labels.append(label_binarizer(amounts[end]))

    return test_data, test_labels


def label_binarizer(amount):
    """
    ワンホットラベルの生成

    Args:
        amount (float): 水位
    Returns:
        array: ワンホットラベル
    """

    if amount <= 10:
        label = 0
    elif 10 < amount and amount <= 20:
        label = 1
    elif 20 < amount and amount <= 30:
        label = 2
    elif 30 < amount and amount <= 40:
        label = 3
    elif 40 < amount and amount <= 50:
        label = 4
    elif 50 < amount and amount <= 60:
        label = 5
    elif 60 < amount and amount <= 70:
        label = 6
    elif 70 < amount and amount <= 80:
        label = 7
    elif 80 < amount and amount <= 90:
        label = 8
    elif 90 < amount and amount <= 100:
        label = 9

    return label


def softmax_to_label(prediction):
    """
    softmax予測値のラベル化

    Args:
        prediction (float): softmax予測
    Returns:
        string: 結果水位
    """

    return np.argmax(prediction)


def get_random_data(mode, data, labels, history):
    """
    ランダムデータの取得

    Args:
        mode (string): train or test
        data (array): データ
        labels (array): ラベル
        history (array): 学習済みデータのインデックス
    Returns:
        array: ランダムデータ
        array: ラベル
    """

    if mode == 'train':
        data_size = BATCH_SIZE
    elif mode == 'test':
        data_size = TEST_ONEFILE_DATA_NUM

    random_data, random_labels = [], []
    while len(random_data) < data_size:
        if len(history) == len(data):
            history = []

        index = random.randint(0, len(data) - 1)
        if index not in history:
            history.append(index)
            random_data.append(data[index])
            random_labels.append(labels[index])

    return random_data, random_labels, history


def main():
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = models.CNN(kernel_size=KERNEL_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)
    softmax = nn.Softmax(dim=1)

    def train():
        """
        モデルの学習
        """

        # データの読み込み
        train_data, train_labels = make_train_data()

        model.train()
        print('\n***** 学習開始 *****')

        history = []
        for epoch in range(EPOCH_NUM):
            # 学習データの作成
            random_data, random_labels, history = get_random_data('train', train_data, train_labels, history)
            # Tensorへ変換
            inputs = torch.tensor(random_data, dtype=torch.float, device=device).view(-1, 1, MFCC_FILTER_NUM)
            labels = torch.tensor(random_labels, dtype=torch.long, device=device)

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

        history = []
        with torch.no_grad():
            # テストデータの作成
            random_data, random_labels, history = get_random_data('test', test_data, test_labels, history)
            # Tensorへ変換
            inputs = torch.tensor(random_data, dtype=torch.float, device=device).view(-1, 1, MFCC_FILTER_NUM)
            labels = torch.tensor(random_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = softmax(outputs)

            labels = labels.to('cpu').detach().numpy().copy()
            outputs = outputs.to('cpu').detach().numpy().copy()
            answers, predictions = [], []
            for label, output in zip(labels, outputs):
                answers.append(label)
                predictions.append(softmax_to_label(output))

            # 結果の記録
            for answer, prediction in zip(answers, predictions):
                answer = str((answer * 10) + 50) + '-' + str(((answer * 10) + 50) + 10)
                prediction = str((prediction * 10) + 50) + '-' + str(((prediction * 10) + 50) + 10)
                result_writer.writerow([TEST_FILE.replace('.', '_'), answer, prediction])
            score = accuracy_score(answers, predictions)
            scores.append(score)
            result_writer.writerow(['(Accuracy)' + TEST_FILE.replace('.', '_'), score])

    # モデルの学習
    loss_all = []
    train()

    # モデルのテスト
    test()

    # Lossの描画
    figures_dir = '../figures/' + now
    if os.path.exists(figures_dir) == False:
        os.makedirs(figures_dir)
    print('\nLossを描画します．．．\n')
    plt.figure(figsize=(16, 9))
    plt.plot(range(EPOCH_NUM), loss_all)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.tick_params(labelsize=26)
    filename = figures_dir + '/' + TEST_FILE.replace('.', '_') + '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # 結果の保存ファイル作成
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = '../data/result2_' + now + '.csv'
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        scores = []
        files = natsorted(glob.glob(SOUND_DIR + '*'))
        for test_index, test_file in enumerate(files):
            # テストデータ以外を学習に使用
            TRAIN_FILES = [os.path.split(filename)[1] for index, filename in enumerate(files) if index != test_index]
            TEST_FILE = os.path.split(test_file)[1]

            # ファイルの検証
            SAMPLING_RATE = get_sampling_rate()
            WINDOW_SIZE = int(WINDOW_SECOND * SAMPLING_RATE)

            print('\n\n----- Test: ' + TEST_FILE.replace('.', '_') + ' -----')
            main()
        result_writer.writerow(['(Avg.)' + BOTTLE, sum(scores) / len(scores)])
