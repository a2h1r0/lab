import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

future_num = 1  # 何日先を予測するか
feature_num = 1
batch_size = 128

time_steps = 30  # lstmのtimesteps
moving_average_num = 30  # 移動平均を取る日数
n_epocs = 5

lstm_hidden_dim = 16
target_dim = 1

path = "./nikkei-225-index-historical-chart-data.csv"

model_name = "./nikkei.mdl"


class LSTM(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(LSTM, self).__init__()
        self.input_dim = lstm_input_dim
        self.hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=1,  # default
                            # dropout=0.2,
                            batch_first=True
                            )
        self.dense = nn.Linear(lstm_hidden_dim, target_dim)

    def forward(self, X_input):
        _, lstm_out = self.lstm(X_input)
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)


def prepare_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num),
                        dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        feats[b_i, :, 0] = X_data[b_idx + 1 - time_steps: b_idx + 1]

    return feats


def Training(search="", wait_load=False):
    print("\nTraining\n")

    # データの作成
    df = pd.read_csv(path, header=8)
    mat = df.query('date.str.match('+search+')', engine='python')
    train_data_t = mat[' value'].values
    labels = mat['date'].values

    train_data = np.arange(len(train_data_t), dtype="float32")

    # 微分(変化量)データに変換
    for i in range(len(train_data_t)-1):
        train_data[i] = train_data_t[i+1]-train_data_t[i]

    # 正規化用ゲイン
    gain = np.max(train_data)-np.min(train_data)
    gain = gain/2

    train_data = train_data/gain  # ±1.0以内に

    X_data, y_data = [], []
    for i in range(len(train_data)-1):
        X_data.append(train_data[i])
        y_data.append(train_data[i+1])

    #データをtrain, testに分割するIndex
    val_idx_from = int(len(train_data)*0.7)
    test_idx_from = int(len(train_data)*0.8)

    # 学習用データ
    X_train, y_train = [], []
    X_train = torch.tensor(X_data[:val_idx_from],
                           dtype=torch.float, device=device)
    y_train = torch.tensor(y_data[:val_idx_from],
                           dtype=torch.float, device=device)

    # 評価用データ
    X_val, y_val = [], []
    X_val = torch.tensor(X_data[val_idx_from:test_idx_from],
                         dtype=torch.float, device=device)
    y_val = y_data[val_idx_from:test_idx_from]

    # テスト用データ
    X_test, y_test = [], []
    X_test = torch.tensor(X_data[test_idx_from:],
                          dtype=torch.float, device=device)
    y_test = y_data[test_idx_from:]

    # モデル定義
    model = LSTM(lstm_input_dim=feature_num,
                 lstm_hidden_dim=lstm_hidden_dim, target_dim=target_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    train_size = X_train.size(0)
    best_acc_score = 0

    for epoch in range(n_epocs):
        # trainデータのindexをランダムに入れ替える。最初のtime_steps分は使わない。
        perm_idx = np.random.permutation(np.arange(time_steps, train_size))
        for t_i in range(0, len(perm_idx), batch_size):
            batch_idx = perm_idx[t_i:(t_i + batch_size)]
            # LSTM入力用の時系列データの準備
            feats = prepare_data(batch_idx, time_steps,
                                 X_train, feature_num, device)
            y_target = y_train[batch_idx]
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
