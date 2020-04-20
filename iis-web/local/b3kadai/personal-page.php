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
    <h1>Personal Page</h1>

    <table>
      <tr>
        <td>
          <ul>
            <li>研究室のホームページでみなさんの氏名，メールアドレスを公表します．</li>
            <li>個人のホームページを作成することを推奨します．</li>
            <ul>
              <li>B4になってからでよいです．</li>
              <li>研究活動は自分の名前を全面に出して行います．</li>
              <li>研究内容や業績を公表してください．</li>
            </ul>
          </ul>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
