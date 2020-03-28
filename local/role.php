<!-- 認証処理 -->
<?php include(__DIR__.'auth.php'); ?>

<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>研究室内限定(Local info)</title>

  <!-- 共有ファイル -->
  <?php include($_SERVER['DOCUMENT_ROOT'].'/include.php'); ?>
</head>

<!-- ヘッダー埋め込み -->
<?php include(__DIR__.'/header.php'); ?>

<body>
  <div class="main">
    <a href="local.php">研究室内限定(Local info)</a>　><br>
<h1>役職 / Role</h1>

<table border="1">
  <caption>2018年度，2019年度</caption>
    <tr>
      <td>レクリエーション</td>
      <td>秋元，八田</td>
    </tr>
    <tr>
      <td>ウェブ(カレンダー) / サーバ</td>
      <td>大山，梁瀬</td>
    </tr>
    <tr>
      <td>会計 / 物品購入</td>
      <td>大木戸，松田</td>
    </tr>
    <tr>
      <td>輪講</td>
      <td>関，野田</td>
    </tr>
    <tr>
      <td>オールマイティ</td>
      <td>藤井，川東</td>
    </tr>
  </table>

  </div>
</body>
</html>
