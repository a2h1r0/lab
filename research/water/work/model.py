import torch.nn as nn


class VGG19(nn.Module):
    """
    VGG19モデル
    """

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_4 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_4 = nn.ReLU()
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc6 = nn.Linear(512, 256)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(256, 256)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(p=0.5)

        self.fc8 = nn.Linear(256, 1)
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=100)

    def forward(self, x):
        """
        Args:
            x (:obj:`Tensor`[batch_size, 1, WINDOW_SIZE]): 音源データ
        Returns:
            :obj:`Tensor`[batch_size, 1]: 識別結果
        """

        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        x = self.pool5(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.fc8(x)
        x = self.hardtanh(x)

        return x


class CNN(nn.Module):
    """
    CNNモデル

    Args:
        kernel_size (int): カーネルサイズ
    """

    def __init__(self, kernel_size=7):
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
