import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from model import Net
import os
os.chdir(os.path.dirname(__file__))


EPOCH_NUM = 10000  # 学習サイクル数
KERNEL_SIZE = 13  # カーネルサイズ（奇数のみ）


def main():
    def train():
        """
        モデルの学習
        """

        model.train()

        '''学習サイクル'''
        for epoch in range(EPOCH_NUM):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(
                    '\nEpoch: {:3d} / Loss: {:.3f}'.format(epoch+1, loss.item()))

    def test():
        """
        モデルのテスト
        """

        model.eval()
        # テスト処理

    # モデルの構築
    model = Net(kernel_size=KERNEL_SIZE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(model.parameters(), lr=0.0002)

    # モデルの学習
    train()

    # モデルのテスト
    test()


if __name__ == '__main__':
    # PyTorchの初期化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)

    main()
