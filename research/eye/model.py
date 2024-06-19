import torch.nn as nn
import numpy as np


class Net(nn.Module):
    """
    CNNモデル

    Args:
        input_size (int): 入力次元数
        output_classes (int): 出力クラス数
    """

    def __init__(self, input_size, output_classes):
        super().__init__()

        map_size = input_size + 6
        self.kernel_size = 5
        self.hidden_size = map_size + 8

        self.conv = nn.Conv1d(
            in_channels=input_size, out_channels=map_size, kernel_size=self.kernel_size)

        self.lstm = nn.LSTM(
            input_size=map_size, hidden_size=self.hidden_size, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, output_classes)

    def forward(self, x, data_length):
        """
        Args:
            x (:obj:`Tensor`[batch_size, FEATURE_SIZE, DATA_LENGTH]): 視線データ
            data_length (array): データ長
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        x = self.conv(x)

        data_length = np.array(data_length) - (self.kernel_size - 1)
        x = nn.utils.rnn.pack_padded_sequence(
            x.permute(0, 2, 1), data_length, batch_first=True, enforce_sorted=False)

        _, x = self.lstm(x)

        x = self.fc(x[0].view(-1, self.hidden_size))

        return x
