import torch
import torch.nn as nn


class Net(nn.Module):
    """
    CNNモデル

    Args:
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, kernel_size=15):
        super().__init__()

        ###--- Encoder ---###
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv4 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv5 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, input):
        """
        Args:
            input (:obj:`Tensor`[1, 2, 256]): 正解 or 生成 色データ + 脈波データのペア
        Returns:
            :obj:`Tensor`[1, 1, 256]: 識別結果
        """

        ###--- Encoder ---###
        conv1_out = self.conv1(input)
        x1 = self.relu1(conv1_out)

        conv2_out = self.conv2(x1)
        x2 = self.relu2(conv2_out)

        conv3_out = self.conv3(x2)
        x3 = self.relu3(conv3_out)

        conv4_out = self.conv4(x3)
        x4 = self.relu4(conv4_out)

        conv5_out = self.conv5(x4)
        x5 = self.relu5(conv5_out)

        # グレースケール化（0 ~ 255）
        out = torch.round(self.hardtanh(conv10_out))

        return out
