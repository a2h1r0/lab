import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

TESTDATA_SIZE = 0.3  # テストデータの割合

EPOCH_NUM = 1000  # 学習サイクル数

WINDOW_SIZE = 10  # ウィンドウサイズ
STEP_SIZE = 5  # ステップ幅

INPUT_DIMENSION = 1  # LSTMの入力次元数（人間の脈波の1次元時系列データ）
HIDDEN_SIZE = 24  # LSTMの隠れ層
OUTPUT_DIMENSION = 1  # LSTMの出力次元数（画面の1次元色データ）

future_num = 1  # 何日先を予測するか
feature_num = 1
batch_size = 128

time_steps = 30  # lstmのtimesteps
moving_average_num = 30  # 移動平均を取る日数
n_epocs = 5

lstm_hidden_dim = 16
target_dim = 1

path = "./work/20200926_233009_fujii.csv"

model_name = "./work/pulse.mdl"


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
            :obj:`Tensor`: 予測結果
        """

        _, lstm_out = self.lstm(input)
        out = self.fc(lstm_out[0].view(-1, self.hidden_size))
        return out


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

    # X = np.reshape(np.array(X), [-1, WINDOW_SIZE, 1])
    # Y = np.reshape(np.array(Y), [-1, 1])
    X = np.reshape(np.array(X), [-1, WINDOW_SIZE])
    Y = np.array(Y)
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


def Training():
    """
    学習
    """

    #--- データの読み込み ---#
    print("Train Data Load Start...")
    df = pd.read_csv(path, header=0)
    human_pulse_time = df['ard_micro'].values
    human_pulse = df['pulse'].values

    X, Y = create_dataset(human_pulse)
    train_x, train_y, test_x, test_y = split_data(X, Y)

    # データの変換
    train_x = torch.tensor(train_x, dtype=torch.float, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)
    test_x = torch.tensor(test_x, dtype=torch.float, device=device)
    test_y = torch.tensor(test_y, dtype=torch.float, device=device)

    #--- 学習開始 ---#
    print("INPUT_DIMENSION:{}\tHIDDEN_SIZE:{}\tOUTPUT_DIMENSION:{}".format(
        INPUT_DIMENSION, HIDDEN_SIZE, OUTPUT_DIMENSION))
    print("Training starts")

    # モデルの定義
    model = LSTM(input_size=INPUT_DIMENSION,
                 hidden_size=HIDDEN_SIZE, output_size=OUTPUT_DIMENSION)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(EPOCH_NUM):
        optimizer.zero_grad()
        # batch size x time steps x feature_num
        # RGBデータ，TKに入れる = model(train_x)
        # # TKの処理
        # # Arduinoの処理
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


def main():
    Training()


if __name__ == '__main__':
    main()
