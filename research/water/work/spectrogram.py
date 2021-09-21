from pydub import AudioSegment
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy
import os
os.chdir(os.path.dirname(__file__))


SOUND_DATA = '../sounds/temp/shampoo/1.mp3'


def calculate_sp(x, n_fft=512, hop_length=256):
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp


def show_sp(sp, fs, hop_length):
    librosa.display.specshow(sp, sr=fs, x_axis="time", y_axis="log", hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def main():
    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DATA, 'mp3')
    fs = len(sound[:1000].get_array_of_samples())
    x = np.array(sound[:500].get_array_of_samples(), dtype=np.float32)

    sp = calculate_sp(x)
    show_sp(sp, fs, hop_length=256)


if __name__ == '__main__':
    main()
