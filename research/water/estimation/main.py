import numpy as np
import pandas as pd
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.optim as optimizers
import model as models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
import scipy
import librosa
import glob
import csv
import datetime
import random
import sys
import os
os.chdir(os.path.dirname(__file__))


SOUND_DIR = '../sounds/'

NUM_CLASSES = 10  # 分類クラス数
EPOCH = 1000  # 学習サイクル数
KERNEL = 3  # カーネルサイズ（奇数のみ）
BATCH = 10000  # バッチサイズ
WINDOW_SECOND = 0.05  # 1サンプルの秒数
STEP_SECOND = 0.02  # スライド幅の秒数
NUM_TEST_ONEFILE_DATA = 1000  # 1ファイルごとのテストデータ数
N_MFCC = 20  # MFCCの次数


def get_sampling_rate():
    """
    サンプリング周波数の取得
    """

    sound = AudioSegment.from_file(SOUND_DIR + TRAIN_FILES[0], 'mp3')

    return len(sound[:1000].get_array_of_samples())


def mfcc(sound_data):
    """
    MFCC

    Args:
        sound_data (:obj:`ndarray`): 音データ
    Returns:
        array: MFCC特徴量配列
    """

    mfccs = librosa.feature.mfcc(sound_data, sr=SAMPLING_RATE, n_mfcc=N_MFCC + 1)
    mfccs = np.delete(mfccs, 0, axis=0)
    feature = np.average(mfccs, axis=1)

    return feature


def make_train_data():
    """
    学習データの作成
    """

    train_data, train_labels = [], []

    for filename in TRAIN_FILES:
        # 音源の読み出し
        sound, _ = librosa.load(SOUND_DIR + filename, sr=SAMPLING_RATE)
        amounts = np.linspace(0, 100, len(sound))

        for index in range(0, len(sound) - WINDOW_SIZE + 1, STEP):
            start = index
            end = start + WINDOW_SIZE - 1
            train_data.append(mfcc(sound[start:end + 1]))
            train_labels.append(amount_to_label(amounts[end]))

    return train_data, train_labels


def make_test_data():
    """
    テストデータの作成
    """

    test_data, test_labels = [], []

    # 音源の読み出し
    sound, _ = librosa.load(SOUND_DIR + TEST_FILE, sr=SAMPLING_RATE)
    amounts = np.linspace(0, 100, len(sound))

    for index in range(0, len(sound) - WINDOW_SIZE + 1, STEP):
        start = index
        end = start + WINDOW_SIZE - 1
        test_data.append(mfcc(sound[start:end + 1]))
        test_labels.append(amount_to_label(amounts[end]))

    return test_data, test_labels


def amount_to_label(amount):
    """
    水位->ラベル

    Args:
        amount (float): 水位
    Returns:
        array: ラベル
    """

    if NUM_CLASSES == 2:
        if amount <= 90:
            label = 0
        elif 90 < amount and amount <= 100:
            label = 1

    elif NUM_CLASSES == 10:
        if amount <= 10:
            label = 0
        elif 10 < amount and amount <= 20:
            label = 1
        elif 20 < amount and amount <= 30:
            label = 2
        elif 30 < amount and amount <= 40:
            label = 3
        elif 40 < amount and amount <= 50:
            label = 4
        elif 50 < amount and amount <= 60:
            label = 5
        elif 60 < amount and amount <= 70:
            label = 6
        elif 70 < amount and amount <= 80:
            label = 7
        elif 80 < amount and amount <= 90:
            label = 8
        elif 90 < amount and amount <= 100:
            label = 9

    return label


def softmax_to_label(prediction):
    """
    softmax予測値のラベル化

    Args:
        prediction (float): softmax予測
    Returns:
        int: 結果ラベル
    """

    return np.argmax(prediction)


def label_to_amount(label):
    """
    ラベル->水位

    Args:
        label (int): ラベル
    Returns:
        string: 水位
    """

    if NUM_CLASSES == 2:
        if label == 0:
            amount = '0-90'
        elif label == 1:
            amount = '90-100'

    elif NUM_CLASSES == 10:
        amount = str(label * 10) + '-' + str((label * 10) + 10)

    return amount


def get_random_data(mode, data, labels, history):
    """
    ランダムデータの取得

    Args:
        mode (string): train or test
        data (array): データ
        labels (array): ラベル
        history (array): 学習済みデータのインデックス
    Returns:
        array: ランダムデータ
        array: ラベル
    """

    if mode == 'train':
        data_size = BATCH
    elif mode == 'test':
        data_size = NUM_TEST_ONEFILE_DATA

    random_data, random_labels = [], []
    while len(random_data) < data_size:
        if len(history) == len(data):
            history = []

        index = random.randint(0, len(data) - 1)
        if index not in history:
            history.append(index)
            random_data.append(data[index])
            random_labels.append(labels[index])

    return random_data, random_labels, history


def main():
    # 初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    # モデルの構築
    model = models.Net(kernel_size=KERNEL, output_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)
    softmax = nn.Softmax(dim=1)

    def train():
        """
        モデルの学習
        """

        # データの読み込み
        train_data, train_labels = make_train_data()

        model.train()
        print('\n***** 学習開始 *****')

        history = []
        for epoch in range(EPOCH):
            # 学習データの作成
            random_data, random_labels, history = get_random_data('train', train_data, train_labels, history)
            # Tensorへ変換
            inputs = torch.tensor(random_data, dtype=torch.float, device=device).view(-1, 1, N_MFCC)
            labels = torch.tensor(random_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if 'coffee' in TEST_FILENAME:
                loss_coffee[-1].append(loss.item())
            elif 'dishwashing' in TEST_FILENAME:
                loss_dishwashing[-1].append(loss.item())
            elif 'shampoo' in TEST_FILENAME:
                loss_shampoo[-1].append(loss.item())
            elif 'skinmilk' in TEST_FILENAME:
                loss_skinmilk[-1].append(loss.item())
            elif 'tokkuri' in TEST_FILENAME:
                loss_tokkuri[-1].append(loss.item())

            if (epoch + 1) % 10 == 0:
                print('Epoch: {} / Loss: {:.3f}'.format(epoch + 1, loss.item()))

        print('\n----- 終了 -----\n')

    def test():
        """
        モデルのテスト
        """

        # データの読み込み
        test_data, test_labels = make_test_data()

        model.eval()
        print('\n***** テスト *****')

        history = []
        with torch.no_grad():
            # テストデータの作成
            random_data, random_labels, history = get_random_data('test', test_data, test_labels, history)
            # Tensorへ変換
            inputs = torch.tensor(random_data, dtype=torch.float, device=device).view(-1, 1, N_MFCC)
            labels = torch.tensor(random_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = softmax(outputs)

            labels = labels.to('cpu').detach().numpy().copy()
            outputs = outputs.to('cpu').detach().numpy().copy()
            answers, predictions = [], []
            for label, output in zip(labels, outputs):
                answers.append(label)
                predictions.append(softmax_to_label(output))

            # 結果の記録
            for answer, prediction in zip(answers, predictions):
                answer_amount = label_to_amount(answer)
                prediction_amount = label_to_amount(prediction)
                result_writer.writerow([TEST_FILENAME, answer_amount, prediction_amount])
                answers_all.append(answer_amount)
                predictions_all.append(prediction_amount)
            score = accuracy_score(answers, predictions)
            result_writer.writerow(['(Accuracy)' + TEST_FILENAME, score])

            if 'coffee' in TEST_FILENAME:
                scores_coffee.append(score)
            elif 'dishwashing' in TEST_FILENAME:
                scores_dishwashing.append(score)
            elif 'shampoo' in TEST_FILENAME:
                scores_shampoo.append(score)
            elif 'skinmilk' in TEST_FILENAME:
                scores_skinmilk.append(score)
            elif 'tokkuri' in TEST_FILENAME:
                scores_tokkuri.append(score)

    train()
    test()


if __name__ == '__main__':
    # 結果の保存ファイル作成
    now = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    result_file = '../data/result_' + str(NUM_CLASSES) + '_classes_' + now + '.csv'
    with open(result_file, 'w', newline='') as f:
        result_writer = csv.writer(f)
        result_writer.writerow(['TestFile', 'Answer', 'Prediction'])

        figures_dir = '../figures/' + str(NUM_CLASSES) + '_classes/' + now
        if os.path.exists(figures_dir) == False:
            os.makedirs(figures_dir)

        loss_coffee, loss_dishwashing, loss_shampoo, loss_skinmilk, loss_tokkuri = [], [], [], [], []
        scores_coffee, scores_dishwashing, scores_shampoo, scores_skinmilk, scores_tokkuri = [], [], [], [], []
        answers_all, predictions_all = [], []
        files = natsorted(glob.glob(SOUND_DIR + '*'))
        if len(files) == 0:
            print('ファイルが存在しません．')
            sys.exit()

        for test_index, test_file in enumerate(files):
            # テストデータ以外を学習に使用
            TRAIN_FILES = [os.path.split(filename)[1] for index, filename in enumerate(files) if index != test_index]
            TEST_FILE = os.path.split(test_file)[1]
            TEST_FILENAME = TEST_FILE.replace('.mp3', '')

            # ファイルの検証
            SAMPLING_RATE = get_sampling_rate()
            WINDOW_SIZE = int(WINDOW_SECOND * SAMPLING_RATE)
            STEP = int(STEP_SECOND * SAMPLING_RATE)

            print('\n\n----- Test: ' + TEST_FILENAME + ' -----')

            if 'coffee' in TEST_FILENAME:
                loss_coffee.append([])
            elif 'dishwashing' in TEST_FILENAME:
                loss_dishwashing.append([])
            elif 'shampoo' in TEST_FILENAME:
                loss_shampoo.append([])
            elif 'skinmilk' in TEST_FILENAME:
                loss_skinmilk.append([])
            elif 'tokkuri' in TEST_FILENAME:
                loss_tokkuri.append([])

            main()

        result_writer.writerow(['(Average)coffee', sum(scores_coffee) / len(scores_coffee)])
        result_writer.writerow(['(Average)dishwashing', sum(scores_dishwashing) / len(scores_dishwashing)])
        result_writer.writerow(['(Average)shampoo', sum(scores_shampoo) / len(scores_shampoo)])
        result_writer.writerow(['(Average)skinmilk', sum(scores_skinmilk) / len(scores_skinmilk)])
        result_writer.writerow(['(Average)tokkuri', sum(scores_tokkuri) / len(scores_tokkuri)])

    # Lossの描画
    print('\nLossを描画します．．．\n')
    plt.figure(figsize=(16, 9))
    plt.plot(range(EPOCH), np.mean(loss_coffee, axis=0), label='Bottle A')
    plt.plot(range(EPOCH), np.mean(loss_dishwashing, axis=0), label='Bottle B')
    plt.plot(range(EPOCH), np.mean(loss_shampoo, axis=0), label='Bottle C')
    plt.plot(range(EPOCH), np.mean(loss_skinmilk, axis=0), label='Bottle D')
    plt.plot(range(EPOCH), np.mean(loss_tokkuri, axis=0), label='Bottle E')
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=26, loc='upper right')
    filename = figures_dir + '/' + TEST_FILENAME + '_loss.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 混同行列の描画
    if NUM_CLASSES == 10:
        scale = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        sns.heatmap(pd.DataFrame(data=confusion_matrix(answers_all, predictions_all),
                                 index=scale, columns=scale), annot=True, cmap='Blues', cbar=False)
        filename = figures_dir + '/confusion_matrix.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
