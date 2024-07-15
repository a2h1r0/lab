import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from model import Net
import matplotlib.pyplot as plt
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


SUBJECTS = ['fujii', 'ueda', 'iguma', 'takeuchi',
            'ogura', 'igarashi', 'kota_sakaguchi']
EXAM_TYPE = 1
DATA_DIR = f'./data/split/window_20/exam_type_{EXAM_TYPE}'


USE_COLUMNS = [
    'device_time_stamp', 'left_gaze_point_in_user_coordinate_system_x', 'left_gaze_point_in_user_coordinate_system_y', 'left_gaze_point_in_user_coordinate_system_z', 'left_pupil_diameter', 'left_gaze_origin_in_user_coordinate_system_x', 'left_gaze_origin_in_user_coordinate_system_y', 'left_gaze_origin_in_user_coordinate_system_z', 'right_gaze_point_in_user_coordinate_system_x', 'right_gaze_point_in_user_coordinate_system_y', 'right_gaze_point_in_user_coordinate_system_z', 'right_pupil_diameter', 'right_gaze_origin_in_user_coordinate_system_x', 'right_gaze_origin_in_user_coordinate_system_y', 'right_gaze_origin_in_user_coordinate_system_z'
]   # 特徴量

# USE_COLUMNS = [
#     'device_time_stamp', 'left_gaze_point_on_display_area_x', 'left_gaze_point_on_display_area_y', 'left_gaze_point_in_user_coordinate_system_x', 'left_gaze_point_in_user_coordinate_system_y', 'left_gaze_point_in_user_coordinate_system_z', 'left_pupil_diameter', 'left_gaze_origin_in_user_coordinate_system_x', 'left_gaze_origin_in_user_coordinate_system_y', 'left_gaze_origin_in_user_coordinate_system_z', 'left_gaze_origin_in_trackbox_coordinate_system_x', 'left_gaze_origin_in_trackbox_coordinate_system_y', 'left_gaze_origin_in_trackbox_coordinate_system_z', 'right_gaze_point_on_display_area_x', 'right_gaze_point_on_display_area_y', 'right_gaze_point_in_user_coordinate_system_x', 'right_gaze_point_in_user_coordinate_system_y', 'right_gaze_point_in_user_coordinate_system_z', 'right_pupil_diameter', 'right_gaze_origin_in_user_coordinate_system_x', 'right_gaze_origin_in_user_coordinate_system_y', 'right_gaze_origin_in_user_coordinate_system_z', 'right_gaze_origin_in_trackbox_coordinate_system_x', 'right_gaze_origin_in_trackbox_coordinate_system_y', 'right_gaze_origin_in_trackbox_coordinate_system_z'
# ]


EPOCH = 1000  # エポック数
NUM_CLASSES = 2  # 分類クラス数


def load_data(subjects):
    """
    データの読み込み

    Returns:
        list: データ
        list: データラベル
        list: データ長
        list: ウィンドウ情報
    """

    def label_to_onehot(label):
        """
        ワンホットラベルの作成

        Args:
            label (string): 読み込んだラベル
        Returns:
            list: ワンホットラベル
        """

        drunk = int('drunk' in label)
        sober = int('sober' in label)

        if NUM_CLASSES == 1:
            label = [drunk]
        elif NUM_CLASSES == 2:
            label = [drunk, sober]

        return label

    data, labels, length, index = [], [], [], []
    files = [filename for filename in glob.glob(
        f'{DATA_DIR}/*/*.csv') if re.search(f'.*({"|".join(subjects)}).*', filename)]

    for filename in files:
        window = pd.read_csv(filename, header=0)
        window_tensor = torch.tensor(
            window.loc[:, USE_COLUMNS].values, dtype=torch.float, device=device)

        data.append(torch.nan_to_num(window_tensor))
        labels.append(label_to_onehot(filename))
        length.append(len(window_tensor))
        index.append({'filename': filename, 'start_timestamp': int(
            window.iloc[0]['device_time_stamp'])})

    return data, labels, length, index


def train(train_subjects):
    """
    モデルの学習

    Args:
        train_subjects (list): 学習被験者
    Returns:
        list: Loss
    """

    # データの読み込み
    train_data, train_labels, train_data_length, _ = load_data(train_subjects)

    model.train()
    print('\n***** 学習開始 *****')

    loss_list = []
    for epoch in range(EPOCH):
        inputs = torch.tensor(rnn.pad_sequence(
            train_data), dtype=torch.float, device=device).view(len(train_data), len(USE_COLUMNS), -1)
        labels = torch.tensor(
            train_labels, dtype=torch.float, device=device)

        optimizer.zero_grad()
        outputs = model(inputs, train_data_length)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

    print('\n----- 終了 -----\n')

    return loss_list


def test(test_subjects):
    """
    モデルのテスト

    Args:
        test_subjects (list): テスト被験者
    Returns:
        list: モデルアウトプット
        list: 正解ラベル
        list: ウィンドウ情報
    """

    def sigmoid_to_onehot(prediction):
        """
        シグモイド予測値のワンホットラベル化

        Args:
            prediction (list): シグモイド予測
        Returns:
            list: ワンホットラベル
        """

        if NUM_CLASSES == 1:
            onehot = list(map(lambda value: int(value > 0.5), prediction))
        elif NUM_CLASSES == 2:
            onehot = list(map(lambda value: int(
                value[0] == np.argmax(prediction)), enumerate(prediction)))

        return onehot

    def onehot_to_label(classes):
        """
        ワンホットラベルのラベル化

        Args:
            classes (list): ワンホットラベル
        Returns:
            int: ラベル
        """

        if NUM_CLASSES == 1:
            label = int(classes[0])
        elif NUM_CLASSES == 2:
            index = list(classes).index(1)
            label = 'drunk' if index == 0 else 'sober'

        return label

    # データの読み込み
    test_data, test_labels, test_data_length, test_index = load_data(
        test_subjects)

    model.eval()
    print('\n***** テスト *****')

    predictions, answers = [], []
    with torch.no_grad():
        inputs = torch.tensor(rnn.pad_sequence(
            test_data), dtype=torch.float, device=device).view(len(test_data), len(USE_COLUMNS), -1)
        labels = torch.tensor(
            test_labels, dtype=torch.float, device=device)

        optimizer.zero_grad()
        outputs = model(inputs, test_data_length)
        outputs = sigmoid(outputs)

        labels = labels.to('cpu').detach().numpy().copy()
        outputs = outputs.to('cpu').detach().numpy().copy()
        for label, output in zip(labels, outputs):
            answers.append(onehot_to_label(label))
            predictions.append(
                [onehot_to_label(sigmoid_to_onehot(output)), output])

    return predictions, answers, test_index


def main(train_subjects, test_subjects, save_dir):
    """
    main function

    Args:
        train_subjects (list): 学習被験者
        test_subjects (list): テスト被験者
        save_dir (string): データ保存先
    Returns:
        list: 正解ラベル
        list: 予測ラベル
    """

    loss_list = train(train_subjects)
    predictions, answers, test_index = test(test_subjects)

    with open(f'{save_dir}/accuracy.csv', 'a', newline='') as f:
        result_writer = csv.writer(f)

        result_writer.writerow(['(train_subjects)'] + train_subjects)
        result_writer.writerow(['(test_subjects)'] + test_subjects)
        result_writer.writerow([])

        result_writer.writerow(
            ['Filename', 'Window', 'Answer', 'Prediction', 'Output'])

        for index, answer, prediction in zip(test_index, answers, predictions):
            result_writer.writerow([index['filename'].split(
                '\\')[-1], index['start_timestamp'], answer, prediction[0], list(prediction[1])])
        result_writer.writerow(['(Accuracy)', accuracy_score(
            answers, [prediction[0] for prediction in predictions])])

        result_writer.writerow([])
        result_writer.writerow(
            ['--------------------------------------------------'])
        result_writer.writerow([])

    # Lossの描画
    print('\nLossを描画します．．．\n')
    plt.figure(figsize=(16, 9))
    plt.plot(range(EPOCH), loss_list)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=26, loc='upper right')
    plt.savefig(
        f'{save_dir}/test_{test_subjects[0]}.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(
        f'{save_dir}/test_{test_subjects[0]}.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    return answers, predictions


def save_total_score(answers, predictions, save_dir):
    """
    最終スコアの保存

    Args:
        answers (list): 正解ラベル
        predictions (list): 予測ラベル
        save_dir (string): データ保存先
    """

    with open(f'{save_dir}/accuracy.csv', 'a', newline='') as f:
        result_writer = csv.writer(f)

        result_writer.writerow(['【Summary】'])
        result_writer.writerow(['(DATA_DIR)', DATA_DIR])
        result_writer.writerow(['(USE_COLUMNS)'] + USE_COLUMNS)
        result_writer.writerow(['(EPOCH)', EPOCH])
        result_writer.writerow(['(Total Accuracy)', accuracy_score(
            answers, [prediction[0] for prediction in predictions])])


if __name__ == '__main__':
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = Net(input_size=len(USE_COLUMNS),
                output_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(model.parameters())
    sigmoid = nn.Sigmoid()

    # 結果の保存
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./result/exam_type_{EXAM_TYPE}/{now}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    answers_all, predictions_all = [], []
    for test_subjects in SUBJECTS:
        train_subjects = [
            subject for subject in SUBJECTS if subject != test_subjects]

        answers, predictions = main(train_subjects, [test_subjects], save_dir)
        answers_all.extend(answers)
        predictions_all.extend(predictions)

    save_total_score(answers_all, predictions_all, save_dir)
