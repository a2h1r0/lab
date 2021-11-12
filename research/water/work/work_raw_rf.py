import numpy as np
from pydub import AudioSegment
import sklearn.ensemble
import sklearn.model_selection
import matplotlib.pyplot as plt
from natsort import natsorted
import glob
import csv
import datetime
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


BOTTLE = 'shampoo'

SOUND_DIR = '../sounds/raw/' + BOTTLE + '/'


WINDOW_SECOND = 0.5  # 1サンプルの秒数
STEP = 500  # スライド幅
TEST_ONEFILE_DATA_NUM = 100  # 1ファイルごとのテストデータ数


def get_sampling_rate():
    """
    サンプリング周波数の取得
    """

    sound = AudioSegment.from_file(SOUND_DIR + TRAIN_FILES[0], 'mp3')

    return len(sound[:1000].get_array_of_samples())


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
        amounts = np.linspace(50, 100, len(data))
        # amounts = np.linspace(0, 100, len(data))

        for index in range(0, len(data) - WINDOW_SIZE + 1, STEP):
            start = index
            end = start + WINDOW_SIZE - 1
            train_data.append(data[start:end + 1])
            train_labels.append(get_label(amounts[end]))

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成
    """

    test_data_all, test_labels_all = [], []

    # 音源の読み出し
    sound = AudioSegment.from_file(SOUND_DIR + TEST_FILE, 'mp3')
    data = np.array(sound.get_array_of_samples())
    data = data[(len(data)//10)*9:]
    # data = data[len(data)//2:]
    amounts = np.linspace(90, 100, len(data))
    # amounts = np.linspace(50, 100, len(data))
    # amounts = np.linspace(0, 100, len(data))

    for index in range(0, len(data) - WINDOW_SIZE + 1, STEP):
        start = index
        end = start + WINDOW_SIZE - 1
        test_data_all.append(data[start:end + 1])
        test_labels_all.append(get_label(amounts[end]))

    test_data, test_labels, history = [], [], []
    while len(test_data) < TEST_ONEFILE_DATA_NUM:
        index = random.randint(0, len(test_data_all) - 1)
        if index not in history:
            history.append(index)
            test_data.append(test_data_all[index])
            test_labels.append(test_labels_all[index])

    return test_data, test_labels


def get_label(amount):
    """
    ラベルの生成

    Args:
        amount (float): 水位
    Returns:
        array: ラベル
    """

    if amount <= 60:
        label = 0
    elif 60 < amount and amount <= 70:
        label = 1
    elif 70 < amount and amount <= 80:
        label = 2
    elif 80 < amount and amount <= 90:
        label = 3
    elif 90 < amount and amount <= 100:
        label = 4

    return label


def prediction_to_label(prediction):
    """
    予測値のラベル化

    Args:
        prediction (float): 予測
    Returns:
        string: 結果水位
    """

    label = (np.argmax(prediction) * 10) + 50

    return str(label) + '-' + str(label + 10)


def main():
    # データの読み込み
    train_data, train_labels = make_train_data()
    test_data, test_labels = make_test_data()

    rf = sklearn.ensemble.RandomForestClassifier()
    rf.fit(train_data, train_labels)

    # 評価
    accuracy = rf.score(test_data, test_labels)
    print('accuracy {0:.2%}'.format(accuracy))

    # 結果のプロット
    outputs = rf.predict(test_data)

    answers, predictions = [], []
    for label, output in zip(labels, outputs):
        answers.append(sigmoid_to_label(label))
        predictions.append(sigmoid_to_label(output))

    # 結果の記録
    for answer, prediction in zip(answers, predictions):
        result_writer.writerow([TEST_FILE.replace('.', '_'), answer, prediction])
    score = accuracy_score(answers, predictions)
    scores.append(score)
    result_writer.writerow(['(Accuracy)' + TEST_FILE.replace('.', '_'), score])


if __name__ == '__main__':
    # 結果の保存ファイル作成
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = '../data/result_raw_' + now + '.csv'
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        scores = []
        files = natsorted(glob.glob(SOUND_DIR + '*'))[:2]
        for test_index, test_file in enumerate(files):
            # テストデータ以外を学習に使用
            TRAIN_FILES = [os.path.split(filename)[1] for index, filename in enumerate(files) if index != test_index]
            TEST_FILE = os.path.split(test_file)[1]

            # ファイルの検証
            SAMPLING_RATE = get_sampling_rate()
            WINDOW_SIZE = int(WINDOW_SECOND * SAMPLING_RATE)

            print('\nTest: ' + TEST_FILE.replace('.', '_'))
            main()
        result_writer.writerow(['(Avg.)' + BOTTLE, sum(scores) / len(scores)])
