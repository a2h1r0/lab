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


DATA_DIR = '../dataset/train/speed/1_13/'

USE_MARKERS = ['right_shoulder', 'right_elbow', 'right_wrist',
               'left_shoulder', 'left_elbow', 'left_wrist']

FEATURE_SIZE = 21  # 特徴量次元数
NUM_CLASSES_MACRO = 5  # マクロクラス数
NUM_CLASSES_MICRO = 2  # マイクロクラス数
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

    train_data, train_labels_macro, train_labels_micro = [], [], []
    files = glob.glob(DATA_DIR + '/subject_[' + ''.join(TRAIN_SUBJECTS) + ']*.csv')
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            raw_data = [row for row in reader]
            feature_data = make_feature(raw_data, USE_MARKERS)
            if len(feature_data[0]) < 5:
                continue
        train_data.append(torch.tensor(feature_data, dtype=torch.float, device=device))
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
        test_data.append(torch.tensor(feature_data, dtype=torch.float, device=device))
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
    def train():
        """
        モデルの学習
        """

        # データの作成
        train_data = get_marker_data(marker, train_data_all)
        train_data_length = [len(data) for data in train_data]

        model.Macro.train()
        model.Micro.train()
        print('\n***** 学習開始 *****')

        for epoch in range(EPOCH_NUM):
            # パディング処理
            inputs = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True).permute(0, 2, 1).to(device)

            # macro識別
            labels_macro = torch.tensor(train_labels_macro, dtype=torch.float, device=device)
            optimizer_macro.zero_grad()
            outputs_macro = model.Macro(inputs, train_data_length)
            loss_macro = criterion_macro(outputs_macro, labels_macro)
            loss_macro.backward()
            optimizer_macro.step()

            # micro識別
            labels_micro = torch.tensor(train_labels_micro, dtype=torch.float, device=device)
            optimizer_micro.zero_grad()
            outputs_micro = model.Micro(inputs, train_data_length)
            loss_micro = criterion_micro(outputs_micro, labels_micro)
            loss_micro.backward()
            optimizer_micro.step()

            loss_macro_all[-1].append(loss_macro.item())
            loss_micro_all[-1].append(loss_micro.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss macro: {:.3f} / Loss micro: {:.3f}'.format(epoch + 1, loss_macro.item(), loss_micro.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        # データの作成
        test_data = get_marker_data(marker, test_data_all)
        test_data_length = [len(data) for data in test_data]

        model.Macro.eval()
        model.Micro.eval()
        print('\n***** テスト *****')

        with torch.no_grad():
            # パディング処理
            inputs = torch.nn.utils.rnn.pad_sequence(test_data, batch_first=True).permute(0, 2, 1).to(device)

            # macro識別
            outputs_macro = model.Macro(inputs, test_data_length)
            prediction_macro = torch.sigmoid(outputs_macro)
            prediction_macro = prediction_macro.to('cpu').detach().numpy().copy()

            # micro識別
            outputs_micro = model.Micro(inputs, test_data_length)
            prediction_micro = torch.sigmoid(outputs_micro)
            prediction_micro = prediction_micro.to('cpu').detach().numpy().copy()

            predictions.append(get_10_prediction(prediction_macro, prediction_micro))

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

    # モデルの構築
    model = Net(input_size=FEATURE_SIZE, hidden_size=HIDDEN_SIZE,
                out_features_macro=NUM_CLASSES_MACRO, out_features_micro=NUM_CLASSES_MICRO, device=device)
    pos_weight_macro = torch.ones([NUM_CLASSES_MACRO], device=device)
    criterion_macro = nn.BCEWithLogitsLoss(pos_weight=pos_weight_macro)
    optimizer_macro = optimizers.Adam(model.Macro.parameters())
    pos_weight_micro = torch.ones([NUM_CLASSES_MICRO], device=device)
    criterion_micro = nn.BCEWithLogitsLoss(pos_weight=pos_weight_micro)
    optimizer_micro = optimizers.Adam(model.Micro.parameters())

    # データの読み込み
    train_data_all, train_labels_macro, train_labels_micro = make_train_data()
    test_data_all, answer_labels, answer_labels_macro, answer_labels_micro = make_test_data()

    loss_macro_all, loss_micro_all = [], []
    predictions = []
    for marker in range(len(USE_MARKERS)):
        print('\n!!!!! ' + USE_MARKERS[marker] + ' !!!!!')

        # モデルの学習
        loss_macro_all.append([])
        loss_micro_all.append([])
        train()

        # モデルのテスト
        test()

    use_files = 'train' + ''.join(TRAIN_SUBJECTS) + '_test' + TEST_SUBJECT
    # 部位ごとの結果の保存
    data_dir = '../data/' + now + '/'
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)
    for marker, prediction_single in zip(USE_MARKERS, predictions):
        prediction_labels_single = [sigmoid_to_label(prediction) for prediction in prediction_single]
        report_df = pd.DataFrame(classification_report(answer_labels, prediction_labels_single, output_dict=True))
        report_df.to_csv(data_dir + 'report_' + marker + '_' + use_files + '.csv')

    # 予測ラベルの決定
    prediction_labels = label_determination(predictions)

    # 全体の結果の保存
    report_df = pd.DataFrame(classification_report(answer_labels, prediction_labels, output_dict=True))
    report_df.to_csv(data_dir + 'report_all.csv')
    print(report_df)
    loss_macro_file = data_dir + 'loss_macro_' + use_files + '.csv'
    with open(loss_macro_file, 'w', newline='') as f:
        loss_macro_writer = csv.writer(f)
        loss_macro_writer.writerow(['Epoch'] + USE_MARKERS)
        for epoch, loss in enumerate(np.array(loss_macro_all).T):
            loss_macro_writer.writerow([epoch + 1] + list(loss))
    loss_micro_file = data_dir + 'loss_micro_' + use_files + '.csv'
    with open(loss_micro_file, 'w', newline='') as f:
        loss_micro_writer = csv.writer(f)
        loss_micro_writer.writerow(['Epoch'] + USE_MARKERS)
        for epoch, loss in enumerate(np.array(loss_micro_all).T):
            loss_micro_writer.writerow([epoch + 1] + list(loss))

    # 結果の描画
    figures_dir = '../figures/' + now + '/'
    if os.path.exists(figures_dir) == False:
        os.makedirs(figures_dir)
    print('\n結果を描画します．．．')
    plt.figure()
    sns.heatmap(confusion_matrix(answer_labels, prediction_labels))
    plt.savefig(figures_dir + 'result_' + use_files + '.png', bbox_inches='tight', pad_inches=0)

    # Lossの描画
    plt.figure(figsize=(16, 9))
    for marker, loss_macro, loss_micro in zip(USE_MARKERS, loss_macro_all, loss_micro_all):
        plt.plot(range(1, EPOCH_NUM + 1), loss_macro, linestyle='solid', label=marker)
        plt.plot(range(1, EPOCH_NUM + 1), loss_micro, linestyle='dashed', label=marker)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.legend()
    plt.tick_params(labelsize=26)
    plt.savefig(figures_dir + 'loss_' + use_files + '.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    TRAIN_SUBJECTS = ['1', '2', '3']
    TEST_SUBJECT = '3'
    main()

    TRAIN_SUBJECTS = ['1', '2', '3']
    TEST_SUBJECT = '2'
    main()

    TRAIN_SUBJECTS = ['1', '2', '3']
    TEST_SUBJECT = '1'
    main()
