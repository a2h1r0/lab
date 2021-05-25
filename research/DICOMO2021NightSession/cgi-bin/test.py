import sys
import json
import sleep


def main():
    # Ajaxデータの受け取り
    key = sys.stdin.read()

    if key in ['cast', 'stop', 'catch']:
        do_fishing_action(key)
    elif key in ['up', 'down', 'left', 'right']:
        move_location(key)


def do_fishing_action(key):
    """釣り動作の実行
    Args:
        key (string): アプリ上で入力されたキー
    """

    # Arduinoに接続してリールを制御
    sleep(3)
    print('Content-type: application/json\n\n' + key)


def move_location(key):
    """竿の移動
    Args:
        key (string): アプリ上で入力されたキー
    """

    # 竿を動かしてもらう処理
    print('Content-type: application/json\n\n' + key)


if __name__ == '__main__':
    main()
