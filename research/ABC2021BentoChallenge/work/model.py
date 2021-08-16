import torch.nn as nn
import numpy as np


class Net():
    """
    識別モデル

    Args:
        input_size (int): 入力次元数
        hidden_size (int): 隠れ層数
        out_features_macro (int): マクロ出力次元数
        out_features_micro (int): マイクロ出力次元数
        device (string): 使用デバイス
    """

    def __init__(self, input_size, hidden_size, out_features_macro, out_features_micro, device='cpu'):
        super().__init__()

        self.Macro = self.Macro(input_size=input_size, hidden_size=hidden_size, out_features=out_features_macro).to(device)
        self.Micro = self.Micro(input_size=input_size, hidden_size=hidden_size, out_features=out_features_micro).to(device)

    class Macro(nn.Module):
        """
        Macro識別モデル

        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層数
            out_features (int): 出力次元数
        """

        def __init__(self, input_size, hidden_size, out_features):
            super().__init__()

            MAP_SIZE = input_size * 6
            self.kernel_size = 5
            self.hidden_size = hidden_size

            self.conv = nn.Conv1d(in_channels=input_size, out_channels=MAP_SIZE, kernel_size=self.kernel_size, groups=input_size)
            self.lstm = nn.LSTM(input_size=MAP_SIZE, hidden_size=self.hidden_size, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, out_features)

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

    class Micro(nn.Module):
        """
        Micro識別モデル

        Args:
            input_size (int): 入力次元数
            hidden_size (int): 隠れ層数
            out_features (int): 出力次元数
        """

        def __init__(self, input_size, hidden_size, out_features):
            super().__init__()

            MAP_SIZE = input_size * 6
            self.kernel_size = 5
            self.hidden_size = hidden_size

            self.conv = nn.Conv1d(in_channels=input_size, out_channels=MAP_SIZE, kernel_size=self.kernel_size, groups=input_size)
            self.lstm = nn.LSTM(input_size=MAP_SIZE, hidden_size=self.hidden_size, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, out_features)

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


class NetAll(nn.Module):
    """
    識別モデル

    Args:
        input_size (int): 入力次元数
        hidden_size (int): 隠れ層数
        out_features (int): 出力次元数
    """

    def __init__(self, input_size, hidden_size, out_features):
        super().__init__()

        MAP_SIZE = input_size * 6
        self.kernel_size = 5
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=MAP_SIZE, kernel_size=self.kernel_size, groups=input_size)
        self.lstm = nn.LSTM(input_size=MAP_SIZE, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, out_features)

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
