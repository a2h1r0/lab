import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import Net
from preprocess import make_feature
from label_determination import majority_vote_sigmoid
import matplotlib.pyplot as plt
from natsort import natsorted
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


ACCELERATION_DIR = '../acceleration/'

FEATURE_SIZE = 21  # 特徴量次元数
NUM_CLASSES = 4  # 分類クラス数
# A：1.0-
# B：0.7-0.9
# C：0.3-0.6
# D：-0.2

EPOCH_NUM = 5000  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数
LABEL_THRESHOLD = 0.0  # ラベルを有効にする閾値


def make_train_data():
    """
    学習データの作成

    Returns:
        array: 学習データ
        array: 学習データラベル
    """

    train_data, train_labels = [], []
    files = glob.glob(
        DATA_DIR + '/subject_[' + ''.join(TRAIN_SUBJECTS) + ']*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
            if len(feature_data[0]) < 5:
                continue
        train_data.append(torch.tensor(
            feature_data, dtype=torch.float, device=device))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
        train_labels_macro.append(multi_label_binarizer_macro(label))
        train_labels_micro.append(multi_label_binarizer_micro(label))

    return train_data, train_labels_macro, train_labels_micro


def make_test_data():
    """
    テストデータの作成

    Returns:
        array: テストデータ
        array: テストデータラベル
        array: テストデータ生ラベル
    """

    test_data, answer_labels, answer_labels_macro, answer_labels_micro = [], [], [], []
    files = glob.glob(DATA_DIR + '/subject_' + TEST_SUBJECT + '*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
            if len(feature_data[0]) < 5:
                continue
        test_data.append(torch.tensor(
            feature_data, dtype=torch.float, device=device))
        activity = re.findall(r'activity_\d+', filename)[0]
        label = int(activity.split('_')[1])
        answer_labels.append(label)
        answer_labels_macro.append(multi_label_binarizer_macro(label))
        answer_labels_micro.append(multi_label_binarizer_micro(label))

    return test_data, answer_labels, answer_labels_macro, answer_labels_micro


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


def multi_label_binarizer_macro(label):
    """
    ラベルの5クラスワンホット化

    Args:
        label (int): ラベル
    Returns:
        array: ワンホットラベル
    """

    y = [0 for i in range(5)]
    y[math.ceil(label/2) - 1] = 1

    return y


def multi_label_binarizer_micro(label):
    """
    ラベルの2クラスワンホット化

    Args:
        label (int): ラベル
    Returns:
        array: ワンホットラベル
    """

    y = [0 for i in range(2)]
    y[~(label % 2)] = 1

    return y


def get_10_prediction(prediction_macro, prediction_micro):
    """
    マクロ予測とマイクロ予測の合算

    Args:
        prediction_macro (array): マクロ予測
        prediction_micro (array): マイクロ予測
    Returns:
        array: 10ラベル予測
    """

    predictions = []
    for macro_labels, micro_labels in zip(prediction_macro, prediction_micro):
        prediction = []
        for macro in macro_labels:
            for micro in micro_labels:
                prediction.append(macro * micro)

        predictions.append(prediction)

    return predictions


def sigmoid_to_label(prediction):
    """
    シグモイド予測値のラベル化

    Args:
        label (int): シグモイド予測
    Returns:
        array: 結果ラベル
    """

    return np.argmax(prediction) + 1


def main():
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = Net(input_size=FEATURE_SIZE, hidden_size=HIDDEN_SIZE,
                out_features_macro=NUM_CLASSES_MACRO, out_features_micro=NUM_CLASSES_MICRO, device=device)
    pos_weight = torch.ones([NUM_CLASSES_MACRO], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optimizers.Adam(model.Macro.parameters())

    def train():
        """
        モデルの学習
        """

        print('\n***** 学習開始 *****')
        model.train()

        for epoch in range(EPOCH_NUM):
            labels = torch.tensor(
                train_labels, dtype=torch.float, device=device)
            optimizer.zero_grad()
            outputs = model(inputs, train_data_length)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_all[-1].append(loss.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(
                    epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        print('\n***** テスト *****')
        model.eval()

        with torch.no_grad():
            outputs = model(inputs, test_data_length)
            prediction = torch.sigmoid(outputs)
            predictions.append(
                prediction.to('cpu').detach().numpy().copy())

    def label_determination(predictions):
        """
        ラベルのワンホット化

        Args:
            predictions (array): 部位ごとの予測
        Returns:
            array: 予測結果
        """

        predictions = np.array(predictions).transpose(1, 0, 2)
        labels = []
        for prediction in predictions:
            labels.append(majority_vote_sigmoid(prediction, LABEL_THRESHOLD))

        return labels

    # データの読み込み
    train_data = make_train_data()
    test_data, answer_labels = make_test_data()

    loss = []
    predictions = []

    loss.append([])
    train()

    test()

    # 予測ラベルの決定
    prediction_labels = label_determination(predictions)

    # 全体の結果の保存
    report_df = pd.DataFrame(classification_report(
        answer_labels, prediction_labels, output_dict=True))
    report_df.to_csv(data_dir + 'report_all_' + subjects + '.csv')
    print(report_df)
    loss_macro_file = data_dir + 'loss_macro_' + subjects + '.csv'
    with open(loss_macro_file, 'w', newline='') as f:
        loss_macro_writer = csv.writer(f)
        loss_macro_writer.writerow(['Epoch'] + USE_MARKERS)
        for epoch, loss in enumerate(np.array(loss_macro_all).T):
            loss_macro_writer.writerow([epoch + 1] + list(loss))
    loss_micro_file = data_dir + 'loss_micro_' + subjects + '.csv'
    with open(loss_micro_file, 'w', newline='') as f:
        loss_micro_writer = csv.writer(f)
        loss_micro_writer.writerow(['Epoch'] + USE_MARKERS)
        for epoch, loss in enumerate(np.array(loss_micro_all).T):
            loss_micro_writer.writerow([epoch + 1] + list(loss))

    # 結果の描画
    figures_dir = '../figures/LSTM2_2/' + now + '/'
    if os.path.exists(figures_dir) == False:
        os.makedirs(figures_dir)
    print('\n結果を描画します．．．')
    plt.figure()
    sns.heatmap(confusion_matrix(answer_labels, prediction_labels))
    plt.savefig(figures_dir + 'result_' + subjects +
                '.png', bbox_inches='tight', pad_inches=0)

    # Lossの描画
    plt.figure(figsize=(16, 9))
    for marker, loss_macro, loss_micro in zip(USE_MARKERS, loss_macro_all, loss_micro_all):
        plt.plot(range(1, EPOCH_NUM + 1), loss_macro,
                 linestyle='solid', label=marker)
        plt.plot(range(1, EPOCH_NUM + 1), loss_micro,
                 linestyle='dashed', label=marker)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.legend()
    plt.tick_params(labelsize=26)
    plt.savefig(figures_dir + 'loss_' + subjects +
                '.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    # 結果の保存ファイル作成
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = '../data/result_' + now + '.csv'
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        TRAIN_SUBJECTS = ['1', '2']
        TEST_SUBJECTS = ['3']
        main()

        # このへん考える
        # result_writer.writerow(['(Avg.)' + BOTTLE, sum(scores) / len(scores)])
