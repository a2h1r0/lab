import numpy as np
from pydub import AudioSegment
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


BOTTLE = 'shampoo'

SOUND_DIR = '../sounds/temp/' + BOTTLE + '/'


WINDOW_SECOND = 0.5  # 1サンプルの秒数
STEP = 10000  # スライド幅
TEST_ONEFILE_DATA_NUM = 10  # 1ファイルごとのテストデータ数

MFCC_FILTER_NUM = 20
MFCC_DIMENSION_NUM = 12


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
    # データの読み込み
    train_data, train_labels = make_train_data()
    test_data_all, test_labels_all = make_test_data()
    test_data, answers, _ = get_random_data('test', test_data_all, test_labels_all, [])

    predictions = []
    for test in test_data:
        min_distance = float('inf')
        for index, train in enumerate(train_data):
            distance = np.linalg.norm(train - test, ord=1)

            if distance < min_distance:
                min_distance = distance
                min_index = index

        predictions.append(train_labels[min_index])

    # 予測と正解の差の合計を計算
    diffs = np.abs(np.array(answers) - np.array(predictions))
    diff = np.sum(diffs) / len(diffs)

    # 結果の表示
    for answer, prediction in zip(answers, predictions):
        print('Answer: {:.3f} / Prediction: {:.3f}'.format(answer, prediction))
        log_writer.writerow([TEST_FILE.replace('.', '_'), answer, prediction])
    print('\nDiff: {:.3f}\n'.format(diff))
    result_writer.writerow([TEST_FILE.replace('.', '_'), diff])
    diff_all.append(diff)


if __name__ == '__main__':
    # 結果の保存ファイル作成
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = '../data/result_mfcc_l1_' + now + '.csv'
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Diff'])

        # 予測値の保存ファイル作成（検証用）
        log_file = '../data/outputs_mfcc_l1_' + now + '.csv'
        with open(log_file, 'w', newline='') as f:
            log_writer = csv.writer(f)
            log_writer.writerow(['TestFile', 'Answer', 'Prediction'])

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
