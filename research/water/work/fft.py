import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import csv
import sys
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = '../sounds/temp/shampoo/1.mp3'

START = 1  # 開始（秒）
END = 2  # 終了（秒）


def fft(sound):
    """
    FFT

    Args:
        sound (:obj:`Sound`): 音オブジェクト
    Returns:
        array: 周波数
        array: パワースペクトル
    """

    start = START * 1000
    end = END * 1000
    if end > len(sound.get_array_of_samples()):
        end = sound.get_array_of_samples()

    sound_data = np.array(sound[start:end].get_array_of_samples())

    # 周波数の取得
    freqs = np.fft.fftfreq(len(sound_data), d=1.0/len(sound[:1000].get_array_of_samples()))
    index = np.argsort(freqs)
    # 振幅スペクトル
    fft = np.abs(np.fft.fft(sound_data))
    # パワースペクトル
    power = fft ** 2

    # 並び替え
    return freqs[index], power[index]


def main():
    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DATA, 'mp3')
    freqs, power = fft(sound)

    plt.figure(figsize=(16, 9))
    plt.plot(freqs, power)
    plt.xlim(0, max(freqs))
    plt.xlabel('Frequency', fontsize=26)
    plt.ylabel('Power', fontsize=26)
    plt.tick_params(labelsize=26)
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
