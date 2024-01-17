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

        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=12, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
            in_channels=12, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_classes)

    def forward(self, x):
        """
        Args:
            x (:obj:`Tensor`[batch_size, FEATURE_SIZE, DATA_LENGTH]): 音源データ
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
