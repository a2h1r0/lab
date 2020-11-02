import sys
import tkinter as tk
import time


# def main():
#     root = tk.Tk()
#     root.title("configure method")
#     root.geometry('300x200')
#     root.label = tk.Label(root, text="Text")

#     root.button = tk.Button(root,
#                             text="Click to change text below",
#                             command=changeText(root))
#     root.button.pack()
#     root.label.pack()
#     root.mainloop()


# def changeText(root):
#     # root.label['text'] = "Text updated"
#     root.label.configure(text="Text Updated")


# if __name__ == '__main__':
#     main()

# class Test():
#     def __init__(self):
#         self.root = tk.Tk()
#         self.label = tk.Label(self.root, text="Text")

#         self.button = tk.Button(self.root,
#                                 text="Click to change text below",
#                                 command=self.changeText)
#         self.button.pack()
#         self.label.pack()
#         self.root.mainloop()
#         time.sleep(10)
#         self.label.configure(text="aaa")

#     def changeText(self):
#         self.label.configure(text="Text Updated")


# app = Test()


import sys
import tkinter
from PIL import Image, ImageTk
import threading
import time


def show():

    # 外から触れるようにグローバル変数で定義
    global item, canvas

    root = tkinter.Tk()
    root.title('test')
    root.geometry("400x300")
    canvas = tkinter.Canvas(bg="black", width=400, height=300)
    root.mainloop()


# スレッドを立ててtkinterの画像表示を開始する
thread1 = threading.Thread(target=show)
thread1.start()

time.sleep(1)  # 3秒毎に切り替え
canvas = tkinter.Canvas(bg="red", width=400, height=300)
canvas.update()
# 切り替えたい画像を定義
# img2 = Image.open('your_image2.jpg')
# img2 = ImageTk.PhotoImage(img2)

# itemを差し替え
# canvas.itemconfig(item, image=img2)
# time.sleep(3)

# # itemをもとに戻す
# img = Image.open('your_image.jpg')
# img = ImageTk.PhotoImage(img)
# canvas.itemconfig(item, image=img)
