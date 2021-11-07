import torch.nn as nn


class CNN(nn.Module):
    """
    CNNモデル

    Args:
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        """
        Args:
            x (:obj:`Tensor`[batch_size, 1, WINDOW_SIZE]): 音源データ
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DCNN(nn.Module):
    """
    CNNモデル

    Args:
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=4, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
            in_channels=4, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.conv3 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.conv4 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.conv5 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

        self.hardtanh = nn.Hardtanh(min_val=0, max_val=100)

    def forward(self, x):
        """
        Args:
            x (:obj:`Tensor`[batch_size, 1, WINDOW_SIZE]): 音源データ
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.hardtanh(x)

        return x


class CNN2D(nn.Module):
    """
    2次元CNNモデル
    Args:
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

        self.hardtanh = nn.Hardtanh(min_val=0, max_val=100)

    def forward(self, x):
        """
        Args:
            x (:obj:`Tensor`[batch_size, 1, WINDOW_SIZE]): 音源データ
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.hardtanh(x)

        return x
