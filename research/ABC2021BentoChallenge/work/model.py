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

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`[batch_size, sequence_length, feature_dimension]): 行動データ
        Returns:
            :obj:`Tensor`[batch_size, 部位数（？）, label_size]: 識別結果
        """

        _, x = self.lstm(input)

        x = self.fc(x[0].view(-1, self.hidden_size))

        return x
