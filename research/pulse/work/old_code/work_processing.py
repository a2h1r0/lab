import socket

host = "192.168.100.40"  # Processingで立ち上げたサーバのIPアドレス
port = 10001  # Processingで設定したポート番号

if __name__ == '__main__':
    socket_client = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)  # オブジェクトの作成
    socket_client.connect((host, port))  # サーバに接続

    socket_client.send('255'.encode('utf-8'))  # データを送信 Python3
