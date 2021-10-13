import numpy as np
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.optim as optimizers
import model as models
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


EPOCH_NUM = 1000  # 学習サイクル数
KERNEL_SIZE = 3  # カーネルサイズ（奇数のみ）
BATCH_SIZE = 10000  # バッチサイズ
WINDOW_SECOND = 0.2  # 1サンプルの秒数
STEP = 10000  # スライド幅
TEST_ONEFILE_DATA_NUM = 1000  # 1ファイルごとのテストデータ数

MFCC_FILTER_NUM = 256
MFCC_DIMENSION_NUM = 128


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

    sampling_rate = get_sampling_rate()

    hanning_x = np.hanning(len(sound_data)) * sound_data
    fft = np.fft.fft(hanning_x)
    amplitude_spectrum = np.abs(fft)
    amplitude_spectrum = amplitude_spectrum[:len(sound_data)//2]
    mel_filter_bank = librosa.filters.mel(sr=sampling_rate, n_fft=len(sound_data) - 1, n_mels=MFCC_FILTER_NUM, fmax=sampling_rate // 2)
    mel_amplitude_spectrum = np.dot(mel_filter_bank, amplitude_spectrum)
    mel_log_power_spectrum = 20 * np.log10(mel_amplitude_spectrum)
    mfcc = scipy.fftpack.dct(mel_log_power_spectrum, norm='ortho')
    mfcc = mfcc[:MFCC_DIMENSION_NUM]
    mel_log_power_spectrum_envelope = scipy.fftpack.idct(mfcc, n=MFCC_FILTER_NUM, norm='ortho')

    return mel_log_power_spectrum_envelope


def make_train_data():
    """
    学習データの作成
    """

    train_data, train_labels = [], []

    for filename in TRAIN_FILES:
        # 音源の読み出し
        sound = AudioSegment.from_file(SOUND_DIR + filename, 'mp3')
        data = np.array(sound.get_array_of_samples())
        data = data[len(data)//2:]
        labels = np.linspace(50, 100, len(data))
        # labels = np.linspace(0, 100, len(data))

        for index in range(0, len(data) - WINDOW_SIZE + 1, STEP):
            start = index
            end = start + WINDOW_SIZE - 1
            train_data.append(mfcc(data[start:end + 1]))
            train_labels.append(labels[end])

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成
    """

    test_data, test_labels = [], []

    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DIR + TEST_FILE, 'mp3')
    data = np.array(sound.get_array_of_samples())
    data = data[len(data)//2:]
    labels = np.linspace(50, 100, len(data))
    # labels = np.linspace(0, 100, len(data))

    for index in range(0, len(data) - WINDOW_SIZE + 1, STEP):
        start = index
        end = start + WINDOW_SIZE - 1
        test_data.append(mfcc(data[start:end + 1]))
        test_labels.append(labels[end])

    return test_data, test_labels


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
    model = models.DCNN(kernel_size=KERNEL_SIZE).to(device)
    # model = models.CNN(kernel_size=KERNEL_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)

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
            labels = torch.tensor(random_labels, dtype=torch.float, device=device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

                # 予測値の保存（検証用）
                if (epoch + 1) == EPOCH_NUM:
                    answers = labels.to('cpu').detach().numpy().copy()
                    answers = answers.reshape(-1)
                    predictions = outputs.to('cpu').detach().numpy().copy()
                    predictions = predictions.reshape(-1)
                    rows = np.array([[epoch + 1 for i in range(len(answers))], answers, predictions], dtype=int).T
                    rows = np.insert(rows.astype('str'), 0, TEST_FILE.replace('.', '_'), axis=1)
                    log_writer.writerows(rows)

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
            labels = torch.tensor(random_labels, dtype=torch.float, device=device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 結果を整形
            answers = labels.to('cpu').detach().numpy().copy()
            answers = answers.reshape(-1)
            predictions = outputs.to('cpu').detach().numpy().copy()
            predictions = predictions.reshape(-1)

            # 予測と正解の差の合計を計算
            diffs = np.abs(answers - predictions)
            diff = np.sum(diffs) / len(diffs)

            # 結果の表示
            for answer, prediction in zip(answers, predictions):
                print('Answer: {:.3f} / Prediction: {:.3f}'.format(answer, prediction))
            print('\nDiff: {:.3f}\n'.format(diff))
            result_writer.writerow([TEST_FILE.replace('.', '_'), diff])
            diff_all.append(diff)

    # モデルの学習
    loss_all = []
    train()

    # モデルのテスト
    test()

    # Lossの描画
    figures_dir = '../figures/mfcc_cnn_' + now
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
    result_file = '../data/result_mfcc_cnn_' + now + '.csv'
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Diff'])

        # 予測値の保存ファイル作成（検証用）
        log_file = '../data/outputs_mfcc_cnn_' + now + '.csv'
        with open(log_file, 'w', newline='') as f:
            log_writer = csv.writer(f)
            log_writer.writerow(['TestFile', 'Epoch', 'Answer', 'Prediction'])

            diff_all = []
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
            result_writer.writerow(['(Avg.)' + BOTTLE, np.average(diff_all)])
