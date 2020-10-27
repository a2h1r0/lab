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
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, lstm_out = self.lstm(input)
        out = self.fc(lstm_out[0].view(-1, self.hidden_size))
        return out


def prepare_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num),
                        dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        feats[b_i, :, 0] = X_data[b_idx + 1 - time_steps: b_idx + 1]

    return feats


def create_dataset(dataset):
    X, Y = [], []
    for i in range(0, len(dataset)-WINDOW_SIZE, STEP_SIZE):
        X.append(dataset[i:i+WINDOW_SIZE])
        Y.append(dataset[i+WINDOW_SIZE])
    X = np.reshape(np.array(X), [-1, WINDOW_SIZE, 1])
    Y = np.reshape(np.array(Y), [-1, 1])
    return X, Y


def split_data(x, y):
    pos = round(len(x) * (1 - TESTDATA_SIZE))
    trainX, trainY = x[:pos], y[:pos]
    testX, testY = x[pos:], y[pos:]
    return trainX, trainY, testX, testY


def Training(search="", wait_load=False):
    #--- データの読み込み ---#
    print("Train Data Load Start...")
    df = pd.read_csv(path, header=0)
    human_pulse_time = df['ard_micro'].values
    human_pulse = df['pulse'].values

    X, Y = create_dataset(human_pulse)
    trainX, trainY, testX, testY = split_data(X, Y)

    # データの変換
    trainX = torch.tensor(trainX, dtype=torch.float, device=device)
    trainY = torch.tensor(trainY, dtype=torch.float, device=device)
    testX = torch.tensor(testX, dtype=torch.float, device=device)
    testY = torch.tensor(testY, dtype=torch.float, device=device)

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
        train_scores = model(feats)
        loss_train = criterion(train_scores, y_target.view(-1, 1))
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
    Training(search="\"^(20)\"", wait_load=False)


if __name__ == '__main__':
    main()
