from pydub import AudioSegment
import argparse
import urllib.request
import pathlib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = '../sounds/temp/shampoo/1.mp3'


def extract_mel(audio, sr, n_mels=64):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels).T
    return mel


def main():
    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DATA, 'mp3')
    fs = len(sound[:1000].get_array_of_samples())
    x = np.array(sound.get_array_of_samples(), dtype=np.float32)

    data = np.log(extract_mel(x, fs))

    plt.figure(figsize=(16, 9))
    plt.plot(range(len(data)), data)
    # plt.xlabel('Epoch', fontsize=26)
    # plt.ylabel('Loss', fontsize=26)
    plt.tick_params(labelsize=26)
    plt.show()


if __name__ == '__main__':
    main()
