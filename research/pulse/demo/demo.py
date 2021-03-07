import numpy as np
from time import sleep
import serial
import socket
from collections import deque
from scipy import signal
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QGraphicsProxyWidget, QStatusBar
import sys
import os
os.chdir(os.path.dirname(__file__))


USB_PORT = 'COM3'

SOCKET_ADDRESS = '192.168.11.2'  # Processingサーバのアドレス
SOCKET_PORT = 10000  # Processingサーバのポート

SAMPLE_SIZE = 500  # 描画するサンプル長


class PlotGraph:
    def __init__(self):
        self.timestamp = deque(maxlen=SAMPLE_SIZE)
        self.raw = deque(maxlen=SAMPLE_SIZE)
        self.generated = deque(maxlen=SAMPLE_SIZE)

        #*** ウィンドウの設定 ***#
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('demo')

        #*** グラフの設定 ***#
        self.plt = self.win.addPlot()
        # タイトルの設定
        title_css = '<style> p { font-size: 60px; color: "#FFFFFF"; } </style>'
        self.plt.setTitle(title_css + '<p>Pulse Value</p>')
        # 軸メモリの設定
        font = QtGui.QFont()
        font.setPixelSize(20)
        self.plt.getAxis("left").setStyle(tickFont=font, tickTextOffset=20)
        self.plt.getAxis("bottom").setStyle(tickFont=font, tickTextOffset=20)
        # 描画範囲の設定
        self.plt.setXRange(0, SAMPLE_SIZE)
        self.plt.setYRange(200, 1100)
        # 凡例の設定
        self.plt.addLegend(offset=(10, 50))
        legend_css = '<style> p { font-size: 55px; color: "#FFFFFF"; } </style>'

        #*** 描画する線の設定 ***#
        self.raw_line = self.plt.plot(
            pen=pg.mkPen(color='r', width=3, antialias=True),
            name=legend_css + '<p>Raw</p>')
        self.generated_line = self.plt.plot(
            pen=pg.mkPen(color='b', width=3, antialias=True),
            name=legend_css + '<p>Generated</p>')

        #*** 心拍数表示バーの設定 ***#
        self.heart_rate = QtGui.QStatusBar()
        self.heart_rate.setStyleSheet('QStatusBar { font-size:50px; }')
        status_bar = QGraphicsProxyWidget()
        status_bar.setWidget(self.heart_rate)
        self.layout = self.win.addLayout(row=0, col=0)
        self.layout.addItem(status_bar)

        #*** 描画サイクルの設定 ***#
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self):
        def make_display_data(pulse):
            """色データの生成

            Args:
                pulse (int): 脈波値
            Returns:
                int: 色データ
            """

            return pulse / 1000 * 10 + 122

        def get_heart_rate():
            """心拍数の計算

            Returns:
                tuple: (生脈波の心拍数, 生成脈波の心拍数)
            """

            # 取得時間
            get_time = (self.timestamp[-1] - self.timestamp[0]) / 1000000

            # 生脈波の心拍数の計算
            raw_peaks, _ = signal.find_peaks(self.raw, distance=30)
            raw_heart_rate = int(len(raw_peaks) / get_time * 60)
            # 生成脈波の心拍数の計算
            generated_peaks, _ = signal.find_peaks(self.generated, distance=30)
            generated_heart_rate = int(len(generated_peaks) / get_time * 60)

            return raw_heart_rate, generated_heart_rate

        # 脈波値の受信
        read_data = ser.readline().rstrip().decode(encoding='UTF-8')
        data = read_data.split(',')

        # 正常値が受信できていることを確認
        if len(data) == 3 and data[0].isdecimal() and data[1].isdecimal() and data[2].isdecimal():
            if len(data[1]) <= 3 and len(data[2]) <= 3:
                self.timestamp.append(int(data[0]))
                self.raw.append(int(data[1]))
                self.generated.append(int(data[2]))

                # 色データの描画
                color = make_display_data(int(data[1]))
                socket_client.send((str(color) + '\0').encode('UTF-8'))
                socket_client.recv(1)

                # グラフの再描画
                self.raw_line.setData(self.raw)
                self.generated_line.setData(self.generated)
                # 心拍数の更新
                if len(self.timestamp) >= 2:
                    raw_heart_rate, generated_heart_rate = get_heart_rate()
                    self.heart_rate.showMessage(
                        'Raw: ' + str(raw_heart_rate) + ' / Generated: ' + str(generated_heart_rate))


if __name__ == '__main__':
    print('\n初期化中...')

    # シリアル通信（Arduino）の初期化
    ser = serial.Serial(USB_PORT, 9600)
    ser.reset_input_buffer()
    sleep(3)  # ポート準備に3秒待機**これがないとシリアル通信がうまく動かない**

    # ソケット通信（Processing）の初期化
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_client.connect((SOCKET_ADDRESS, SOCKET_PORT))

    # 描画の開始
    graph = PlotGraph()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    # シリアル通信の終了
    ser.close()
