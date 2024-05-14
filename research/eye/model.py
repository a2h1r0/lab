import torch.nn as nn


class Net(nn.Module):
    """
    CNNモデル

    Args:
        input_size (int): 入力次元数
        output_classes (int): 出力クラス数
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, input_size, output_classes, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_size, out_channels=24, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        # このへんの次元数の整形する
        self.lstm = nn.LSTM(
            input_size=24, hidden_size=32, batch_first=True)

        self.fc = nn.Linear(32, output_classes)

    def forward(self, x):
        """
        Args:
            x (:obj:`Tensor`[batch_size, FEATURE_SIZE, DATA_LENGTH]): 視線データ
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        x = self.conv(x)
        x = self.lstm(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
