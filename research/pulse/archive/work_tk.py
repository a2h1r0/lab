import tkinter
import time
import threading


class Display:
    """
    ディスプレイ表示制御クラス
    """

    def start(self, color):
        """ディスプレイの表示開始

        引数で渡された色で点灯する．

        Args:
            color (string): 表示色
        """

        global thread
        # スレッドの立ち上げ
        thread = threading.Thread(target=self.tk_show, args=(color,))
        thread.start()

    def exit(self):
        """
        ディスプレイの表示終了
        """

        # 処理終了
        self.running = False
        # スレッドの終了
        thread.join()

    def tk_show(self, color):
        """Tkinterの処理

        Args:
            color (string): 表示色
        """

        # 処理開始
        self.running = True
        # 定義
        self.window = tkinter.Tk()
        self.window.geometry('2000x2000')
        self.window.attributes("-topmost", True)
        self.window.configure(background=color)
        # 終了チェックの割り込み
        self.window.after(100, self._check_to_quit)
        # メインループ
        self.window.mainloop()
        # 後処理
        del self.window

    def _check_to_quit(self):
        """
        Tkinterの終了確認
        """

        if self.running:
            # 終了確認の継続
            self.window.after(100, self._check_to_quit)
        else:
            # メインループの停止
            self.window.destroy()


def main():
    display = Display()

    display.start('blue')
    # 処理
    time.sleep(0.5)
    display.exit()

    display.start('red')
    # 処理
    time.sleep(0.5)
    display.exit()


if __name__ == '__main__':
    main()
