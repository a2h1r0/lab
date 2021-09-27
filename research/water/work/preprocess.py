from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob
import os
os.chdir(os.path.dirname(__file__))


SOUND_DIR = '../sounds/raw/shampoo/'  # 音源データ
TEMP_DIR = '../sounds/temp/shampoo/'  # 保存先
THRESHOLD = -55  # 無音検出閾値
SOUND_LENGTH = 3  # 音源ファイルの最短時間


def preprocess(filename):
    """
    音源データ前処理（無音部分の削除）

    Args:
        filename (string): 処理する音源ファイル名
    """

    if os.path.exists(TEMP_DIR) == False:
        os.makedirs(TEMP_DIR)

    data = AudioSegment.from_file(filename, 'mp3')
    sounds = split_on_silence(data, min_silence_len=2000, silence_thresh=THRESHOLD)
    for i in range(len(sounds)):
        sound = sounds[i].set_channels(1)
        if len(sound) > SOUND_LENGTH*1000:
            break

    sound.export(TEMP_DIR + filename.split('\\')[1].split('.')[0] + '.mp3', format='mp3')


if __name__ == '__main__':
    files = glob.glob(SOUND_DIR + '*.mp3')
    for filename in files:
        preprocess(filename)
