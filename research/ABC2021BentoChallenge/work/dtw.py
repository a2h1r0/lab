import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from preprocess import make_raw
from natsort import natsorted
import statistics
import csv
import glob
import re
import datetime
import sys
import os
os.chdir(os.path.dirname(__file__))


DATA_DIR = '../dataset/train/autocorrelation/'

USE_MARKERS = ['right_shoulder', 'right_elbow', 'right_wrist',
               'left_shoulder', 'left_elbow', 'left_wrist']


def make_train_data():
    """
    学習データの作成

    Returns:
        array: 学習データ
        array: 学習データラベル
        array: 学習データファイル
    """

    train_data, labels = [], []
    files = glob.glob(DATA_DIR + '/subject_[' + ''.join(TRAIN_SUBJECTS) + ']*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_raw(raw_data, USE_MARKERS)
        wave_data = []
        for single_data in feature_data:
            wave = []
            for data in single_data:
                wave.append(np.sum(data))
            wave_data.append(wave)
        train_data.append(wave_data)
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
        labels.append(label)
        if len(train_data) > 2:
            break

    return train_data, labels


def make_test_data():
    """
    テストデータの作成

    Returns:
        array: テストデータ
        array: テストデータラベル
        array: テストデータ生ラベル
    """

    test_data, answer_labels = [], []
    files = glob.glob(DATA_DIR + '/subject_' + TEST_SUBJECT + '*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_raw(raw_data, USE_MARKERS)
        wave_data = []
        for single_data in feature_data:
            wave = []
            for data in single_data:
                wave.append(np.sum(data))
            wave_data.append(wave)
        test_data.append(wave_data)
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
        answer_labels.append(label)
        if len(test_data) > 2:
            break

    return test_data, answer_labels


def get_marker_data(marker_index, data):
    """
    部位ごとのデータの取得

    Args:
        marker_index (int): 使用する部位のインデックス
        data (array): データ
    Returns:
        array: 部位ごとのデータ
    """

    return [row[marker_index] for row in data]


def main():
    # データの読み込み
    train_data_all, labels = make_train_data()
    test_data_all, answer_labels = make_test_data()

    predictions = []
    for test_data in test_data_all:
        prediction_labels, prediction_distances = [], []
        for marker in range(len(test_data)):
            test_data_single = test_data[marker]
            train_data_singles = get_marker_data(marker, train_data_all)
            distances = []
            for train_data_single in train_data_singles:
                distance, path = fastdtw(train_data_single, test_data_single, dist=euclidean)
                distances.append(distance)

            min_index = np.argmin(distances)
            prediction_labels.append(labels[min_index])
            prediction_distances.append(distances[min_index])

        majority_labels = statistics.multimode(prediction_labels)
        if len(majority_labels) == 1:
            label = majority_labels[0]
        else:
            min_distance_index = np.argmin(prediction_distances)
            label = prediction_labels[min_distance_index]

        predictions.append(label)

    sys.exit()
    # データの作成
    test_data = get_marker_data(marker, test_data_all)

    loss_all = []
    predictions = []
    for marker in range(len(USE_MARKERS)):
        print('\n!!!!! ' + USE_MARKERS[marker] + ' !!!!!')

        # モデルの学習
        loss_all.append([])
        train()

        # モデルのテスト
        test()

    subjects = 'train' + ''.join(TRAIN_SUBJECTS) + '_test' + TEST_SUBJECT

    # 結果の保存
    data_dir = '../data/LSTM1/' + now + '/'
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)
    report_df = pd.DataFrame(classification_report(answer_labels, prediction_labels, output_dict=True))
    report_df.to_csv(data_dir + 'report_all_' + subjects + '.csv')
    print(report_df)


if __name__ == '__main__':
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

    TRAIN_SUBJECTS = ['1']
    TEST_SUBJECT = '2'
    main()
    TEST_SUBJECT = '3'
    main()

    TRAIN_SUBJECTS = ['2']
    TEST_SUBJECT = '1'
    main()
    TEST_SUBJECT = '3'
    main()

    TRAIN_SUBJECTS = ['3']
    TEST_SUBJECT = '1'
    main()
    TEST_SUBJECT = '2'
    main()
