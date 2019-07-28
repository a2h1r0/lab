import serial
from time import sleep
from matplotlib import pyplot as plt

Sensors = 20            # センサ数の設定
Range = int(Sensors/2)  # 2台に分かれるのでループ範囲は半分
t = [0] * 100           # 時間の範囲

# Arduino1号機
ser = serial.Serial('COM5', 9600)
y = [[0] * 100 for i in range(Range)]   # 1センサにつき100個の値，これをセンサ数分定義
line = [0] * Range                      # センサ数分のグラフ線を定義
# Arduino2号機
ser2 = serial.Serial('COM6', 9600)
y2 = [[0] * 100 for i in range(Range)]
line2 = [0] * Range

sleep(1)    # ポート準備に1秒待機**これがないとシリアル通信がうまく動かない**

# グラフの体裁調整
fig, ax = plt.subplots(Sensors, 1, figsize=(8,40), sharex='all')
plt.subplots_adjust(hspace=0.6)
plt.ion()   # リアルタイム描画の際でもコマンドを受け付けるようにする

# グラフの描画(初期化)
for i in range(Range):
    line[i], = ax[i].plot(t, y[i])          # lineはax[0]~ax[Range-1]
    line2[i], = ax[Range+i].plot(t, y2[i])  # line2はax[Range]~ax[Sensors-1]
for i in range(Sensors):
    ax[i].set_ylim(0, 5)                # 全てのグラフの範囲を0V~5Vに設定
ax[Sensors-1].set_xlabel("time[s]")     # 一番下
ax[Range].set_ylabel("Voltage[V]")      #　中央付近

# シリアル通信から起動時間の取得，Arduino2機の平均を基準時間とする
ser.write("!".encode('UTF-8'))
data = ser.readline().rstrip().split()
ser2.write("?".encode('UTF-8'))
data2 = ser2.readline().rstrip().split()
time_base = (float(data[0])+float(data2[0]))/2  # 先頭要素が時間

# グラフ描画ループ
while True:
    try:
        # シリアル通信とデータ加工
        ser.write("!".encode('UTF-8'))
        data = ser.readline().rstrip().split()
        ser2.write("?".encode('UTF-8'))
        data2 = ser2.readline().rstrip().split()
        
        # 時間管理(x値設定)
        time = (float(data[0])+float(data2[0]))/2
        t.append((time-time_base)/10**6)   # 起動からの経過時間を配列に追加
        del t[0]                           # キューとして扱い，先頭の古い時間値を削除
        ax[0].set_xlim(min(t), max(t))     # 1つのx軸の範囲を更新，他のx軸は使い回し
        
        # y値設定後，全てのaxの値を更新していく
        for i in range(Range):
            # センサごとに100個のy値を持った配列を生成
            # Arduino1号機に接続されているセンサの値
            y[i].append(float(data[i+1]))   #　y配列にy値を追加
            del y[i][0]                     # キューとして先頭のy値を削除
            line[i].set_xdata(t)            # 描画するx値の設定
            line[i].set_ydata(y[i])         # 描画するy値の設定
            # Arduino2号機に接続されているセンサの値
            y2[i].append(float(data2[i+1]))
            del y2[i][0]
            line2[i].set_xdata(t)
            line2[i].set_ydata(y2[i])

        plt.pause(0.01)  # 再描画までに一瞬間隔を空ける**これがないとグラフがフリーズする**
        plt.draw()

    except KeyboardInterrupt:
        break
    
ser.close()
ser2.close()