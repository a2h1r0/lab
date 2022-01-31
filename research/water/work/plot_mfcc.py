import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import os
os.chdir(os.path.dirname(__file__))


FILENAME = '../sounds/coffee_1.mp3'

WINDOW_SECOND = 0.2  # 1サンプルの秒数
STEP_SECOND = 0.02  # スライド幅の秒数
N_MFCC = 60  # MFCCの次数


def get_sampling_rate(filename):
    """
    サンプリング周波数の取得

    Args:
        filename (string): ファイル名
    Returns:
        int: サンプリング周波数
    """

    sound = AudioSegment.from_file(filename, 'mp3')

    return len(sound[:1000].get_array_of_samples())


def mfcc(sound_data):
    """
    MFCC

    Args:
        sound_data (:obj:`ndarray`): 音データ
    Returns:
        array: MFCC特徴量配列
    """

    mfccs = librosa.feature.mfcc(sound_data, sr=SAMPLING_RATE, n_mfcc=N_MFCC + 1)
    mfccs = np.delete(mfccs, 0, axis=0)
    feature = np.average(mfccs, axis=1)

    return feature


# ファイルの検証
SAMPLING_RATE = get_sampling_rate(FILENAME)
WINDOW_SIZE = int(WINDOW_SECOND * SAMPLING_RATE)
STEP = int(STEP_SECOND * SAMPLING_RATE)


sound, _ = librosa.load(FILENAME, sr=SAMPLING_RATE)
amounts = np.linspace(0, 100, len(sound))

for index in range(0, len(sound) - WINDOW_SIZE + 1, STEP):
    start = index
    end = start + WINDOW_SIZE - 1
    if index == 70 * STEP:
        x = len(mfcc(sound[start:end + 1]))
        plt.figure(figsize=(16, 9))
        plt.plot(range(x), mfcc(sound[start:end + 1]), color='red')
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.savefig('mfcc_1d.svg', bbox_inches='tight', pad_inches=0)
        plt.show()
