<!-- 認証処理 -->
<?php include(__DIR__.'/auth.php'); ?>

<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>研究室内限定(Local info) - トップ</title>

  <!-- 共有ファイル -->
  <?php include($_SERVER['DOCUMENT_ROOT'].'/include.php'); ?>
  <!-- 個別CSS -->
  <!-- <link rel="stylesheet" type="text/css" href="/css/index.css"> -->
</head>

<!-- ヘッダー埋め込み -->
<?php include(__DIR__.'/header.php'); ?>

<body>
  <div class="main">
<h1>研究室内限定(Local info) - トップ</h1>
<ul>
<li><a href="role.php">役職一覧</a></li>
<li><a href="./b3kadai">B3が配属されてから半年間にやること</a></li>
<li><a href="meeting.php">ミーティング，輪講</a></li>
<li><a href="knowledge.php">研究に必要な知識</a></li>
<li><a href="important-paper.php">重要論文</a></li>
<li><a href="dataset.php">データセット</a></li>
<li><a href="programming.php">プログラム</a></li>
<li><a href="server-network.php">サーバ，ネットワーク関連</a></li>
<li><a href="univ-support.php">学内補助金関係</a></li>
<li><a href="how-to-submit-papers.php">学会で発表するときの流れ</a></li>
<li><a href="buy-items.php">備品購入の方法</a></li>
<li><a href="society-info.php">研究会・国際会議リスト</a></li>
<li><a href="membership.php">学会会員番号・著者紹介</a></li>
<li><a href="murao-memo.php">村尾メモ</a></li>
</ul>

<iframe src="https://calendar.google.com/calendar/embed?src=iis.ise.ritsumei.ac.jp_9kd8knqnlpv1iogtiu68bsi748%40group.calendar.google.com&ctz=Asia%2FTokyo" style="border: 0" width="100%" height="800px" frameborder="0" scrolling="no"></iframe>

  </div>
</body>
</html>
