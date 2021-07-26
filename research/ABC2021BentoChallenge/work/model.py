import torch.nn as nn
import numpy as np


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

        FEATURE_SIZE = 21
        MAP_SIZE = FEATURE_SIZE * 6
        self.kernel_size = 5
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(in_channels=FEATURE_SIZE, out_channels=MAP_SIZE, kernel_size=self.kernel_size, groups=FEATURE_SIZE)
        self.lstm = nn.LSTM(input_size=MAP_SIZE, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x, data_length):
        """
        Args:
            x (:obj:`Tensor`[batch_size, sequence_length, feature_dimension]): 行動データ
            data_length (array): データ長
        Returns:
            :obj:`Tensor`[batch_size, 部位数（？）, label_size]: 識別結果
        """

        x = self.conv(x)

        data_length = np.array(data_length) - (self.kernel_size - 1)
        x = nn.utils.rnn.pack_padded_sequence(x.permute(0, 2, 1), data_length, batch_first=True, enforce_sorted=False)

        _, x = self.lstm(x)

        x = self.fc(x[0].view(-1, self.hidden_size))

        return x
