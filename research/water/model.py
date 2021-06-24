import torch
import torch.nn as nn


class Net(nn.Module):
    """
    CNNモデル

    Args:
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size)

        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.pool2 = nn.MaxPool1d(kernel_size=kernel_size)

        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(32, 1)

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`[batch_size, 1, WINDOW_SIZE]): 音源データ
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        conv1_out = self.conv1(input)
        pool1_out = self.pool1(conv1_out)

        conv2_out = self.conv2(pool1_out)
        pool2_out = self.pool2(conv2_out)

        conv3_out = self.conv3(pool2_out)
        pool3_out = self.pool3(conv3_out)

        pool3_out = pool3_out.view(pool3_out.size(0), -1)
        out = self.fc1(pool3_out)

        return out
