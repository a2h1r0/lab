import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import serial
from time import sleep


TESTDATA_SIZE = 0.3  # テストデータの割合

EPOCH_NUM = 1000  # 学習サイクル数

WINDOW_SIZE = 32  # ウィンドウサイズ
STEP_SIZE = 1  # ステップ幅
BATCH_SIZE = WINDOW_SIZE  # バッチサイズ

VECTOR_SIZE = 1  # 扱うベクトルのサイズ（脈波は1次元）

INPUT_DIMENSION = 1  # LSTMの入力次元数（人間の脈波の1次元時系列データ）
HIDDEN_SIZE = 24  # LSTMの隠れ層
OUTPUT_DIMENSION = 1  # LSTMの出力次元数（画面の1次元色データ）

DATA_FILE = "./research/pulse/work/2sec.csv"

USB_PORT = "COM3"


class LSTM(nn.Module):
    """
    LSTMモデル
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層サイズ
            output_size (int): 出力サイズ
        """

        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`): 学習データ

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        _, lstm_out = self.lstm(input)
        linear_out = self.fc(lstm_out[0].view(-1, self.hidden_size))
        out = torch.sigmoid(linear_out)

        # 色データの生成
        color_data = self.convert_to_color_data(out)

        # 予測結果の色データを出力
        return color_data

    def convert_to_color_data(self, out):
        """色データへの変換

        予測結果から色データを生成する．

        Args:
            out (:obj:`Tensor`): 予測結果

        Returns:
            :obj:`Numpy`: 予測結果の色データ
        """

        # 色の最大値は16進数FFFF
        COLOR_MAX = 65535

        # Tensorから1次元のNumpyへ
        out = out.detach().cpu().numpy().reshape(-1)

        # 出力の最大値を色の最大値に合わせる（整数）
        converted_data = np.array(out * COLOR_MAX, dtype=int)

        return converted_data


def create_dataset(dataset):
    """データセットの作成

    入力データと，それに対する正解データを作成．

    Args:
        dataset (array): ファイルから読み込んだ時系列データ

    Returns:
        array: 入力データ
        array: 正解データ
    """

    X, Y = [], []
    for i in range(0, len(dataset) - WINDOW_SIZE, STEP_SIZE):
        # ウィンドウサイズ単位の配列
        X.append(dataset[i:i + WINDOW_SIZE])
        # X終了の次のデータ（過去の時系列データから未来を予測）
        Y.append(dataset[i + WINDOW_SIZE])

    # LSTMに入力可能な形式に変換
    # Sequence_Length x Batch_Size x Vector_Size
    X = np.reshape(np.array(X), [-1, BATCH_SIZE, VECTOR_SIZE])
    # Sequence_Length x Vector_Size
    Y = np.reshape(np.array(Y), [-1, VECTOR_SIZE])

    return X, Y


def split_data(x, y):
    """データセットの分割

    学習データと，テストデータに分割．

    Args:
        x (array): 入力データ
        y (array): 正解データ

    Returns:
        array: 学習入力データ
        array: 学習正解データ
        array: テスト入力データ
        array: テスト正解データ
    """

    pos = round(len(x) * (1 - TESTDATA_SIZE))
    train_x, train_y = x[:pos], y[:pos]
    test_x, test_y = x[pos:], y[pos:]

    return train_x, train_y, test_x, test_y


def send_color_and_get_pulse(color):
    """色データの送信と脈波の取得

    色データを送信し，ディスプレイに描画．
    その後，脈波センサからデータを取得して出力．

    Args:
        color (int): 色データ

    Returns:
        pulse_value (int): 脈波値
    """

    # 末尾に終了文字を追加
    send_data = str(color) + '\0'

    # 色データの送信
    ser.write(send_data.encode('UTF-8'))

    # 脈波値の受信
    pulse_value = ser.readline().rstrip().decode(encoding="UTF-8")

    return pulse_value


def Training():
    """
    学習
    """

    #!!!--- データの読み込み ---!!!#
    print("\nデータロード中...\n")
    df = pd.read_csv(DATA_FILE, header=0)
    human_pulse_time = df['ard_micro'].values
    human_pulse = df['pulse'].values

    #--- データセットの作成 ---#
    X, Y = create_dataset(human_pulse)
    train_x, train_y, test_x, test_y = split_data(X, Y)

    #--- データの変換 ---#
    train_x = torch.tensor(train_x, dtype=torch.float, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)
    test_x = torch.tensor(test_x, dtype=torch.float, device=device)
    test_y = torch.tensor(test_y, dtype=torch.float, device=device)

    #!!!--- 学習開始 ---!!!#
    print("\n---学習開始!!!---")
    print("INPUT_DIMENSION:{}\tBATCH_SIZE:{}\tHIDDEN_SIZE:{}\tOUTPUT_DIMENSION:{}\n".format(
        INPUT_DIMENSION, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_DIMENSION))

    # モデルの定義
    model = LSTM(input_size=INPUT_DIMENSION,
                 hidden_size=HIDDEN_SIZE, output_size=OUTPUT_DIMENSION)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    optimizer.zero_grad()

    # 学習サイクル
    # 予測値（色データ）の取得
    colors = model(train_x)

    # 予測結果を1件ずつ処理
    pulse_values = []
    for color in colors:
        # ディスプレイの描画と脈波の取得
        pulse_value = send_color_and_get_pulse(color)
        pulse_values.append(pulse_value)

    print(pulse_values)
    # time.sleep(0.5)


'''
    for epoch in range(EPOCH_NUM):
        optimizer.zero_grad()
        # RGBデータ，TKに入れる = model(train_x)
        color = model(train_x)
        display.start(color)

        # Arduinoの取得処理
        # time.sleep(0.5)


        # loss_train = criterion(Arduinoから取得したセンサ値, y_target.view(-1, 1))
        loss_train.backward()
        optimizer.step()

        # validationデータの評価
        print('EPOCH: ', str(epoch), ' loss :', loss_train.item())
        # with torch.no_grad():
        #     feats_val = prepare_data(np.arange(time_steps, X_val.size(
        #         0)), time_steps, X_val, feature_num, device)
        #     val_scores = model(feats_val)
        #     tmp_scores = val_scores.view(-1).to('cpu').numpy()
        #     bi_scores = np.round(tmp_scores)
        #     acc_score = accuracy_score(y_val[time_steps:], bi_scores)
        #     roc_score = roc_auc_score(y_val[time_steps:], tmp_scores)
        #     f1_scores = f1_score(y_val[time_steps:], bi_scores)
        #     print('Val ACC Score :', acc_score, ' ROC AUC Score :',
        #           roc_score, 'f1 Score :', f1_scores)

        # # validationの評価が良ければモデルを保存
        # if acc_score > best_acc_score:
        #     best_acc_score = acc_score
        #     torch.save(model.state_dict(), model_name)
        #     print('best score updated, Pytorch model was saved!!', )

    torch.save(model.state_dict(), model_name)

    # bestモデルで予測する。
    model.load_state_dict(torch.load(model_name))

    with torch.no_grad():
        feats_test = prepare_data(np.arange(time_steps, X_test.size(
            0)), time_steps, X_test, feature_num, device)
        val_scores = model(feats_test)
        val_labels = labels[time_steps:val_scores.size(0)]
        tmp_scores = val_scores.view(-1).to('cpu').numpy()
        predict = tmp_scores * gain

    # plt.plot(np.arange(train_data_t.size), train_data_t)
    plt.plot(np.arange(predict.size), predict)
    plt.show()
'''


def main():
    Training()


if __name__ == '__main__':
    print("\n初期化中...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    # シリアル通信の初期化
    ser = serial.Serial(USB_PORT, 9600)
    ser.reset_input_buffer()
    sleep(3)  # ポート準備に3秒待機**これがないとシリアル通信がうまく動かない**

    # メイン処理
    main()

    # シリアル通信の終了
    ser.close()
