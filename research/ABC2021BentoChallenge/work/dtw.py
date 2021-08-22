import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from preprocess import make_raw
import statistics
import csv
import glob
import re
import datetime
import time
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
    start = time.perf_counter()

    # データの読み込み
    train_data_all, labels = make_train_data()
    test_data_all, answer_labels = make_test_data()

    prediction_labels = []
    for index, test_data in enumerate(test_data_all):
        print('残り{:d}件．．．'.format(len(test_data_all) - index))

        predictions, prediction_distances = [], []
        for marker in range(len(test_data)):
            test_data_single = test_data[marker]
            train_data_singles = get_marker_data(marker, train_data_all)
            distances = []
            for train_data_single in train_data_singles:
                # 波形に1秒以上のデータの差があれば無視
                if abs(len(train_data_single) - len(test_data_single)) > 100:
                    continue
                distance, path = fastdtw(train_data_single, test_data_single, dist=euclidean)
                distances.append(distance)
            if len(distances) == 0:
                # 結果が存在しなかった場合は制限を解除
                for train_data_single in train_data_singles:
                    distance, path = fastdtw(train_data_single, test_data_single, dist=euclidean)
                    distances.append(distance)

            min_index = np.argmin(distances)
            predictions.append(labels[min_index])
            prediction_distances.append(distances[min_index])

        majority_labels = statistics.multimode(predictions)
        if len(majority_labels) == 1:
            label = majority_labels[0]
        else:
            min_distance_index = np.argmin(prediction_distances)
            label = predictions[min_distance_index]

        prediction_labels.append(label)

    finish = time.perf_counter()
    process_time = finish - start
    process_times.append([''.join(TRAIN_SUBJECTS), TEST_SUBJECT, process_time])

    subjects = 'train' + ''.join(TRAIN_SUBJECTS) + '_test' + TEST_SUBJECT

    # 結果の保存
    accuracy_file = data_dir + 'report_' + subjects + '.csv'
    with open(accuracy_file, 'w', newline='') as f:
        accuracy_writer = csv.writer(f)
        accuracy_writer.writerow(['Marker', 'Accuracy'])
        accuracy_writer.writerow(['all', accuracy_score(answer_labels, prediction_labels)])


if __name__ == '__main__':
    # 結果保存ディレクトリの作成
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    data_dir = '../data/' + now + '/'
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)

    process_times = []
    TRAIN_SUBJECTS = ['1']
    TEST_SUBJECT = '2'
    main()
    # TEST_SUBJECT = '3'
    # main()

    # TRAIN_SUBJECTS = ['2']
    # TEST_SUBJECT = '1'
    # main()
    # TEST_SUBJECT = '3'
    # main()

    # TRAIN_SUBJECTS = ['3']
    # TEST_SUBJECT = '1'
    # main()
    # TEST_SUBJECT = '2'
    # main()

    time_file = data_dir + 'prediction_time.csv'
    with open(time_file, 'w', newline='') as f:
        time_writer = csv.writer(f)
        time_writer.writerow(['train', 'test', 'time'])
        time_writer.writerows(process_times)
