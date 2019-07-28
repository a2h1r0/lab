import csv

To = "abc.csv"      # 追記したいファイル(結合後残したいファイル)
From = "def.csv"    # 結合したいファイル(削除したいファイル)

with open(To, 'a', newline='') as w:
    writer = csv.writer(w)
    
    with open(From, 'r', newline='') as r:
        reader = csv.reader(r)
        next(reader)    # ラベル(1行目)を読み飛ばす
        
        for row in reader:          # 1行ずつ最後まで読み出す
            writer.writerow(row)    # 読み出した行を書き込む

print("Finish")