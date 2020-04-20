<!-- 認証処理 -->
<?php include(__DIR__.'/../auth.php'); ?>

<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>研究室内限定(Local info)</title>

  <!-- 共有ファイル -->
  <?php include($_SERVER['DOCUMENT_ROOT'].'/include.php'); ?>
</head>

<!-- ヘッダー埋め込み -->
<?php include(__DIR__.'/../header.php'); ?>

<body>
  <div class="main">
    <a href="../local.php">研究室内限定(Local info)</a>　>　<a href="index.php">B3課題</a>　><br>
    <h1>PCsetup</h1>

    <table>
      <tr>
        <td>
          <ul>
            <li>
              <span>PCの購入</span>
            </li>
            <ul>
              <li>研究室から一人1台貸与します．windowsかmacか選べますが研究室としてはwindows．</li>
              <li>研究室にあるマシンが古い場合（4年程度経っている），新しいものを購入しますので，希望を聞くことがあります．</li>
              <li>必ずパスワードをかけること．</li><li>紛失，破損した場合は必ず教員に伝えること．</li>
            </ul>
            <li>
              <a href="software-to-be-installed.php">ソフトウェアインストール</a>
            </li>
            <ul>
              <li>上記リンクを参照すること．</li>
              <li>正常にインストールとアクティベーションが完了しているか確認すること．</li>
              <li>TeXに関してはOverleafを使っているのでローカルにインストールする必要はなくなってきた．</li>
            </ul>
            <li>
              <a href="mail.pdf">メーラ設定</a>
            </li>
            <ul>
              <li>研究室で作成したメールアドレス宛のメールを読めるようにする．</li>
              <li>普段使っている別のアドレスに転送する設定にしておくのでもよい．（読むだけであればそれで十分）</li>
              <li>上記PDF中の「@ci.ritsumei.ac.jp」とある部分はすべて「@iis.ise.ritsumei.ac.jp」と読み替えてください．</li>
            </ul>
            <li>
              <span>チームドライブ</span>
            </li>
            <ul>
              <li>輪講の資料や各種写真などを入れておく共有ドライブ（G Suite）があります．</li>
              <li>個人用フォルダとチーム用フォルダがあります．容量は無制限です．</li>
              <li>ただし，おけるファイル数，ディレクトリの深さには制限があります．詳しくはググって．</li>
            </ul>
            <li>
              <span>Googleカレンダー</span>
            </li>
            <ul>
              <li>研究室の予定を書き込む用のカレンダーがあります．</li>
              <li>B4はカレンダーの見方をB3に教えること．</li>
            </ul>
            <li>
              <a href="personal-page.php">個人ページ作成</a>
            </li>
            <ul>
              <li>研究室に配属されると自分のホームページを持つ人も多いです．</li>
              <li>研究業績や社会活動などをアピールすることでいろいろ役立ちます．</li>
              <li>研究室のメンバのページからリンクを張ることもできるので，村尾まで．</li>
            </ul>
          </ul>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
