import numpy as np
import pandas as pd
import math
import csv
import glob
import re
import datetime
import random
import copy
import sys
import os
os.chdir(os.path.dirname(__file__))


RAW_DIR = './data/raw/'
SAVE_DIR = './data/preprocess/window_30/'


WINDOW_SIZE = 30  # ウィンドウサイズ（秒）
STEP = 3  # ステップ幅（秒）


def preprocess():
    """
    データの前処理

    Returns:
        list: 前処理データ
        list: ラベル
    """

    train_data, train_labels = [], []
    files = glob.glob(f'{RAW_DIR}/**/*.csv', recursive=True)

    for filename in files:
        save_dir = os.path.dirname(filename).replace(RAW_DIR, SAVE_DIR)

        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            # データファイルの最終行が空行
            raw = [row[1:8] for row in reader][:-1]

        data, labels = slide_window(raw, filename)

        with open(result_file, 'w', newline='') as f:
            result_writer = csv.writer(f)
            result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

            loss_all = train()
            predictions, answers, test_files = test()

            # 結果の保存
            for filename, answer, prediction in zip(test_files, answers, predictions):
                result_writer.writerow(
                    [filename.split('\\')[-1], answer, prediction])
            result_writer.writerow(
                ['(Accuracy)', accuracy_score(answers, predictions)])

        train_data.extend(data)
        train_labels.extend(labels)

    return train_data, train_labels


def slide_window(raw, label):
    """
    スライディングウィンドウの実行

    Args:
        raw (list): ローデータ
        label (list): ラベル
    Returns:
        list: スライディングウィンドウ後のデータ
    """

    data = []
    labels = []

    for start_index, start in enumerate(raw):
        start_time = datetime.datetime.fromisoformat(start[0])
        end_time = start_time + datetime.timedelta(seconds=WINDOW_SIZE)

        window = []
        for index, row in enumerate(raw[start_index:]):
            time = datetime.datetime.fromisoformat(row[0])
            if time > end_time:
                break
            window.append(list(map(lambda value: float(value), row[1:])))
            end_index = start_index + index

        data.append(torch.tensor(window, dtype=torch.float, device=device))
        labels.append(label_to_onehot(label))

        # 末尾到達
        if end_index == len(raw) - 1:
            break

    return data, labels


def main():
    train_data, train_labels = preprocess()

    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = './result/{}.csv'.format(now)
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        loss_all = train()
        predictions, answers, test_files = test()

        # 結果の保存
        for filename, answer, prediction in zip(test_files, answers, predictions):
            result_writer.writerow(
                [filename.split('\\')[-1], answer, prediction])
        result_writer.writerow(
            ['(Accuracy)', accuracy_score(answers, predictions)])


if __name__ == '__main__':
    main()
