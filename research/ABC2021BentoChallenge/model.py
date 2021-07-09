import torch.nn as nn


class Net(nn.Module):
    """
    LSTMモデル

    Args:
        input_size (int): 入力次元数
        hidden_size (int): 隠れ層数
        out_features (int): 出力次元数
    """

    def __init__(self, input_size, hidden_size, out_features):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`[WINDOW_SIZE, batch_size, 入力次元数（部位？）]): 行動データ
        Returns:
            :obj:`Tensor`[batch_size, 部位数（？）, label_size]: 識別結果
        """

        _, lstm_out = self.lstm(input)

        fc_out = self.fc(lstm_out[0].view(input.size(0), -1))
        out = self.sigmoid(fc_out)

        return out
