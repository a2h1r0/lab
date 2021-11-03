import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import csv
import sys
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = '../sounds/temp/shampoo/1.mp3'

# END = 6  # 終了（秒）


def fft(sound, end):
    """
    FFT

    Args:
        end (int): 表示終了秒数
        sound (:obj:`Sound`): 音オブジェクト
    Returns:
        array: 周波数
        array: パワースペクトル
    """

    end *= 1000
    if end > len(sound.get_array_of_samples()):
        end = sound.get_array_of_samples()
    start = end - 1000

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
    plt.figure(figsize=(16, 9))
    plt.xlim(0, 20000)
    plt.ylim(0, 500000000000)
    plt.xlabel('Frequency', fontsize=26)
    plt.ylabel('Power', fontsize=26)
    plt.tick_params(labelsize=26)

    for i in range(6):
        end = i + 1

        # 音源の読み出し
        sound = AudioSegment.from_file(SOUND_DATA, 'mp3')
        freqs, power = fft(sound, end)

        delete_index = np.where(freqs < 0)
        freqs = np.delete(freqs, delete_index)
        power = np.delete(power, delete_index)
        max_power_index = np.argmax(power)

        print('Max Power Frequency: {:f}Hz'.format(freqs[max_power_index]))

        plt.plot(freqs, power, label='{:d}~{:d} [s]'.format(end, end + 1))

    plt.legend()
    # plt.title(str(END) + '~' + str(END + 1) + ' [s]')
    # plt.savefig(str(END) + '.eps', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
