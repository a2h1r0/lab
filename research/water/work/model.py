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
        self.bn1 = nn.BatchNorm1d(8)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.bn2 = nn.BatchNorm1d(16)

        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
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
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.hardtanh(x)

        return x
