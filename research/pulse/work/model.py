import torch
import torch.nn as nn


class Pix2Pix(nn.Module):
    """
    Pix2Pixモデル

    Args:
        kernel_size (int): カーネルサイズ
        device (string): 使用デバイス
    """

    def __init__(self, kernel_size=15, device='cpu'):
        super().__init__()

        self.G = Pix2Pix.Generator(kernel_size=kernel_size).to(device)
        self.D = Pix2Pix.Discriminator(kernel_size=kernel_size).to(device)

    class Generator(nn.Module):
        """
        生成器：脈波データからそれを再現する色データ（グレースケール）を生成

        Args:
            input (:obj:`Tensor`[1, 1, 256]): 脈波データ
        Returns:
            :obj:`Tensor`[1, 1, 256]: inputを再現する色データ
        """

        def __init__(self, kernel_size):
            super().__init__()

            ###--- Encoder ---###
            self.conv1 = nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu3 = nn.ReLU(inplace=True)

            self.conv4 = nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu4 = nn.ReLU(inplace=True)

            self.conv5 = nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu5 = nn.ReLU(inplace=True)

            ###--- Decoder ---###
            self.conv6 = nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu6 = nn.ReLU(inplace=True)

            # Skip Connection (conv4)
            self.conv7 = nn.ConvTranspose1d(
                in_channels=64 * 2, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu7 = nn.ReLU(inplace=True)

            # Skip Connection (conv3)
            self.conv8 = nn.ConvTranspose1d(
                in_channels=32 * 2, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu8 = nn.ReLU(inplace=True)

            # Skip Connection (conv2)
            self.conv9 = nn.ConvTranspose1d(
                in_channels=16 * 2, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu9 = nn.ReLU(inplace=True)

            # Skip Connection (conv1)
            self.conv10 = nn.ConvTranspose1d(
                in_channels=8 * 2, out_channels=1, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

            # グレースケール化（122 ~ 132）
            self.hardtanh = nn.Hardtanh(min_val=122, max_val=132)

        def forward(self, input):
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

            ###--- Decoder ---###
            conv6_out = self.conv6(x5)
            relu6_out = self.relu6(conv6_out)

            # Skip Connection (conv4)
            conv7_out = self.conv7(torch.cat([relu6_out, x4], dim=1))
            relu7_out = self.relu7(conv7_out)

            # Skip Connection (conv3)
            conv8_out = self.conv8(torch.cat([relu7_out, x3], dim=1))
            relu8_out = self.relu8(conv8_out)

            # Skip Connection (conv2)
            conv9_out = self.conv9(torch.cat([relu8_out, x2], dim=1))
            relu9_out = self.relu9(conv9_out)

            # Skip Connection (conv1)
            conv10_out = self.conv10(torch.cat([relu9_out, x1], dim=1))

            # グレースケール化（0 ~ 255）
            out = torch.round(self.hardtanh(conv10_out))

            return out

    class Discriminator(nn.Module):
        """
        識別器：色データと脈波データのペアから識別結果を出力

        Args:
            input (:obj:`Tensor`[1, 2, 256]): 正解 or 生成 色データ + 脈波データのペア
        Returns:
            :obj:`Tensor`[1, 1, 256]: 識別結果
        """

        def __init__(self, kernel_size):
            super().__init__()

            # Colors + Generated Pulse
            self.conv1 = nn.Conv1d(
                in_channels=1 * 2, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu3 = nn.ReLU(inplace=True)

            self.conv4 = nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu4 = nn.ReLU(inplace=True)

            self.conv5 = nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu5 = nn.ReLU(inplace=True)

            self.conv6 = nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu6 = nn.ReLU(inplace=True)

            self.conv7 = nn.Conv1d(
                in_channels=256, out_channels=1, kernel_size=1)

        def forward(self, input):
            # Colors + Generated Pulse
            conv1_out = self.conv1(input)
            relu1_out = self.relu1(conv1_out)

            conv2_out = self.conv2(relu1_out)
            relu2_out = self.relu2(conv2_out)

            conv3_out = self.conv3(relu2_out)
            relu3_out = self.relu3(conv3_out)

            conv4_out = self.conv4(relu3_out)
            relu4_out = self.relu4(conv4_out)

            conv5_out = self.conv5(relu4_out)
            relu5_out = self.relu5(conv5_out)

            conv6_out = self.conv6(relu5_out)
            relu6_out = self.relu6(conv6_out)

            out = self.conv7(relu6_out)

            return out


class GAN(nn.Module):
    """
    GANモデル

    Args:
        kernel_size (int): カーネルサイズ
        device (string): 使用デバイス
    """

    def __init__(self, kernel_size=15, device='cpu'):
        super().__init__()

        self.G = GAN.Generator(kernel_size=kernel_size).to(device)
        self.D = GAN.Discriminator(kernel_size=kernel_size).to(device)

    class Generator(nn.Module):
        """
        生成器：脈波データからそれを再現する色データ（グレースケール）を生成

        Args:
            input (:obj:`Tensor`[1, 1, 256]): 脈波データ
        Returns:
            :obj:`Tensor`[1, 1, 256]: inputを再現する色データ
        """

        def __init__(self, kernel_size):
            super().__init__()

            self.conv1 = nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu3 = nn.ReLU(inplace=True)

            self.conv4 = nn.Conv1d(
                in_channels=32, out_channels=1, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

            # グレースケール化（0 ~ 255）
            self.hardtanh = nn.Hardtanh(min_val=0, max_val=255)

        def forward(self, input):
            conv1_out = self.conv1(input)
            relu1_out = self.relu1(conv1_out)

            conv2_out = self.conv2(relu1_out)
            relu2_out = self.relu2(conv2_out)

            conv3_out = self.conv3(relu2_out)
            relu3_out = self.relu3(conv3_out)

            conv4_out = self.conv4(relu3_out)

            # グレースケール化（0 ~ 255）
            out = torch.round(self.hardtanh(conv4_out))

            return out

    class Discriminator(nn.Module):
        """
        識別器：色データと脈波データのペアから識別結果を出力

        Args:
            input (:obj:`Tensor`[1, 2, 256]): 正解 or 生成 色データ + 脈波データのペア
        Returns:
            :obj:`Tensor`[1, 1, 256]: 識別結果
        """

        def __init__(self, kernel_size):
            super().__init__()

            self.conv1 = nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
            self.relu3 = nn.ReLU(inplace=True)

            self.conv4 = nn.Conv1d(
                in_channels=32, out_channels=1, kernel_size=kernel_size, padding=(kernel_size-1) // 2)

        def forward(self, input):
            conv1_out = self.conv1(input)
            relu1_out = self.relu1(conv1_out)

            conv2_out = self.conv2(relu1_out)
            relu2_out = self.relu2(conv2_out)

            conv3_out = self.conv3(relu2_out)
            relu3_out = self.relu3(conv3_out)

            out = self.conv4(relu3_out)

            return out


class Test(nn.Module):
    """
    Pix2Pixモデル

    Args:
        kernel_size (int): カーネルサイズ
        device (string): 使用デバイス
    """

    def __init__(self, kernel_size=15, device='cpu'):
        super().__init__()

        self.G = Test.Generator(kernel_size=kernel_size).to(device)
        self.D = Test.Discriminator(kernel_size=kernel_size).to(device)

    class Generator(nn.Module):
        """
        生成器：脈波データからそれを再現する色データ（グレースケール）を生成

        Args:
            input (:obj:`Tensor`[1, 1, 256]): 脈波データ
        Returns:
            :obj:`Tensor`[1, 1, 256]: inputを再現する色データ
        """

        def __init__(self, kernel_size):
            super().__init__()

            ###--- Encoder ---###
            self.conv1 = nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=5, padding=2)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU(inplace=True)

            self.conv4 = nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.relu4 = nn.ReLU(inplace=True)

            ###--- Decoder ---###
            self.conv5 = nn.ConvTranspose1d(
                in_channels=64, out_channels=32, kernel_size=3, padding=1)
            self.relu5 = nn.ReLU(inplace=True)

            # Skip Connection (conv3)
            self.conv6 = nn.ConvTranspose1d(
                in_channels=32 * 2, out_channels=16, kernel_size=3, padding=1)
            self.relu6 = nn.ReLU(inplace=True)

            # Skip Connection (conv2)
            self.conv7 = nn.ConvTranspose1d(
                in_channels=16 * 2, out_channels=8, kernel_size=3, padding=1)
            self.relu7 = nn.ReLU(inplace=True)

            # Skip Connection (conv1)
            self.conv8 = nn.ConvTranspose1d(
                in_channels=8 * 2, out_channels=1, kernel_size=5, padding=2)
            self.relu8 = nn.ReLU(inplace=True)

        def forward(self, input):
            ###--- Encoder ---###
            conv1_out = self.conv1(input)
            x1 = self.relu1(conv1_out)

            conv2_out = self.conv2(x1)
            x2 = self.relu2(conv2_out)

            conv3_out = self.conv3(x2)
            x3 = self.relu3(conv3_out)

            conv4_out = self.conv4(x3)
            x4 = self.relu4(conv4_out)

            ###--- Decoder ---###
            conv5_out = self.conv5(x4)
            relu5_out = self.relu5(conv5_out)

            # Skip Connection (conv3)
            conv6_out = self.conv6(torch.cat([relu5_out, x3], dim=1))
            relu6_out = self.relu6(conv6_out)

            # Skip Connection (conv2)
            conv7_out = self.conv7(torch.cat([relu6_out, x2], dim=1))
            relu7_out = self.relu7(conv7_out)

            # Skip Connection (conv1)
            conv8_out = self.conv8(torch.cat([relu7_out, x1], dim=1))
            relu8_out = self.relu8(conv8_out)

            # グレースケール化（0 ~ 255）
            out = self.convert_to_gray_scale(relu8_out)

            return out

        def convert_to_gray_scale(self, out):
            """色データへの変換

            予測結果から色データを生成する．

            Args:
                out (:obj:`Tensor`): 予測結果

            Returns:
                :obj:`Tensor`: 予測結果の色データ
            """

            return torch.round(out / out.max() * 255)

    class Discriminator(nn.Module):
        """
        識別器：色データと脈波データのペアから識別結果を出力

        Args:
            input (:obj:`Tensor`[1, 2, 256]): 正解 or 生成 色データ + 脈波データのペア
        Returns:
            :obj:`Tensor`[1, 1, 256]: 識別結果
        """

        def __init__(self, kernel_size):
            super().__init__()

            # Colors + Generated Pulse
            self.conv1 = nn.Conv1d(
                in_channels=1 * 2, out_channels=8, kernel_size=5, padding=2)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(inplace=True)

            self.conv3 = nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU(inplace=True)

            self.conv4 = nn.Conv1d(
                in_channels=32, out_channels=1, kernel_size=1)

        def forward(self, input):
            # Colors + Generated Pulse
            conv1_out = self.conv1(input)
            relu1_out = self.relu1(conv1_out)

            conv2_out = self.conv2(relu1_out)
            relu2_out = self.relu2(conv2_out)

            conv3_out = self.conv3(relu2_out)
            relu3_out = self.relu3(conv3_out)

            out = self.conv4(relu3_out)

            return out
