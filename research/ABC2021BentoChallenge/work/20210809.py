import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import NetAll
from preprocess import make_feature
from label_determination import majority_vote_sigmoid
import matplotlib.pyplot as plt
from natsort import natsorted
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

NUM_CLASSES = 10  # クラス数
EPOCH_NUM = 5000  # 学習サイクル数
HIDDEN_SIZE = 24  # 隠れ層数
LABEL_THRESHOLD = 0.0  # ラベルを有効にする閾値


def make_train_data():
    """
    学習データの作成

    Returns:
        array: 学習データ
        array: 学習データラベル
        array: 学習データファイル
    """

    train_data, train_labels, train_files = [], [], []
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
        train_labels.append(multi_label_binarizer(label))
        train_files.append(filename.split('\\')[-1])

    return train_data, train_labels, train_files


def make_test_data():
    """
    テストデータの作成

    Returns:
        array: テストデータ
        array: テストデータラベル
        array: テストデータ生ラベル
    """

    test_data, test_labels, answer_labels = [], [], []
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
        test_labels.append(multi_label_binarizer(label))
        answer_labels.append(label)

    return test_data, test_labels, answer_labels


def get_random_data(train_data_all, train_labels_all, train_files_all):
    """
    ランダムデータの作成

    Args:
        train_data_all (array): 学習データ全件
        train_labels_all (array): 学習データラベル全件
        train_files_all (array): 学習データラベル全件
    Returns:
        array: 学習データ
        array: 学習データラベル
        array: 学習データファイル
    """

    train_data = copy.deepcopy(train_data_all)
    train_labels = copy.deepcopy(train_labels_all)
    sample_num = len(train_data_all)
    delete_index = random.sample(range(sample_num), sample_num // 10)

    for index in sorted(delete_index, reverse=True):
        del train_data[index]
        del train_labels[index]

    train_files = [train_file for index, train_file in enumerate(train_files_all) if index not in delete_index]

    return train_data, train_labels, train_files


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


def multi_label_binarizer(label):
    """
    ラベルのワンホット化

    Args:
        label (int): ラベル
    Returns:
        array: ワンホットラベル
    """

    y = [0 for i in range(10)]
    y[label - 1] = 1

    return y


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
        train_data = get_marker_data(marker, train_data_full_markers)
        train_data_length = [len(data) for data in train_data]

        model.train()
        print('\n***** 学習開始 *****')

        for epoch in range(EPOCH_NUM):
            # パディング処理
            inputs = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True).permute(0, 2, 1).to(device)
            labels = torch.tensor(train_labels, dtype=torch.float, device=device)

            optimizer.zero_grad()
            outputs = model(inputs, train_data_length)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_all[-1].append(loss.item())
            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        # データの作成
        test_data = get_marker_data(marker, test_data_full_markers)
        test_data_length = [len(data) for data in test_data]

        model.eval()
        print('\n***** テスト *****')

        with torch.no_grad():
            # パディング処理
            inputs = torch.nn.utils.rnn.pad_sequence(test_data, batch_first=True).permute(0, 2, 1).to(device)
            labels = torch.tensor(test_labels, dtype=torch.float, device=device)

            outputs = model(inputs, test_data_length)
            # 予測結果をSigmoidに通す
            prediction = torch.sigmoid(outputs)
            predictions.append(prediction.to('cpu').detach().numpy().copy())

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
    model = NetAll(input_size=21, hidden_size=HIDDEN_SIZE, out_features=NUM_CLASSES).to(device)
    pos_weight = torch.ones([NUM_CLASSES], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optimizers.Adam(model.parameters())

    # データの読み込み
    train_data_all, train_labels_all, train_files_all = make_train_data()
    test_data_full_markers, test_labels, answer_labels = make_test_data()

    # データの切り捨て
    train_data_full_markers, train_labels, train_files = get_random_data(train_data_all, train_labels_all, train_files_all)

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

    # 部位ごとの結果の保存
    data_dir = '../data/' + now + '/'
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)
    train_file_save = data_dir + 'train_files_' + subjects + '.csv'
    with open(train_file_save, 'w', newline='') as f:
        train_files_writer = csv.writer(f)
        train_files_writer.writerows([[filename] for filename in natsorted(train_files)])

    for marker, prediction_single in zip(USE_MARKERS, predictions):
        prediction_labels_single = [sigmoid_to_label(prediction) for prediction in prediction_single]
        report_df = pd.DataFrame(classification_report(answer_labels, prediction_labels_single, output_dict=True))
        report_df.to_csv(data_dir + 'report_' + marker + '_' + subjects + '.csv')

    # 予測ラベルの決定
    prediction_labels = label_determination(predictions)

    # 全体の結果の保存
    report_df = pd.DataFrame(classification_report(answer_labels, prediction_labels, output_dict=True))
    report_df.to_csv(data_dir + 'report_all_' + subjects + '.csv')
    print(report_df)
    loss_file = data_dir + 'loss_' + subjects + '.csv'
    with open(loss_file, 'w', newline='') as f:
        loss_writer = csv.writer(f)
        loss_writer.writerow(['Epoch'] + USE_MARKERS)

        for epoch, loss in enumerate(np.array(loss_all).T):
            loss_writer.writerow([epoch + 1] + list(loss))

    # 結果の描画
    figures_dir = '../figures/' + now + '/'
    if os.path.exists(figures_dir) == False:
        os.makedirs(figures_dir)
    print('\n結果を描画します．．．')
    plt.figure()
    sns.heatmap(confusion_matrix(answer_labels, prediction_labels))
    plt.savefig(figures_dir + 'result_' + subjects + '.png', bbox_inches='tight', pad_inches=0)

    # Lossの描画
    plt.figure(figsize=(16, 9))
    for marker, loss in zip(USE_MARKERS, loss_all):
        plt.plot(range(1, EPOCH_NUM + 1), loss, label=marker)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.legend(fontsize=26, loc='upper right')
    plt.tick_params(labelsize=26)
    plt.savefig(figures_dir + 'loss_' + subjects + '.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    TRAIN_SUBJECTS = ['1', '2', '3']
    TEST_SUBJECT = '3'
    main()

    # TRAIN_SUBJECTS = ['1', '3']
    TEST_SUBJECT = '2'
    main()

    # TRAIN_SUBJECTS = ['2', '3']
    TEST_SUBJECT = '1'
    main()
