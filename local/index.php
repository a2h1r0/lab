<!-- 認証処理 -->
<?php
session_start();

$error_message = "";

if(isset($_POST["login"])) {
	if($_POST["password"] == "029D!SK!") {
		$_SESSION["password"] = $_POST["password"];
		header("Location: local.php");
		exit;
	}

	$error_message = "※パスワードが間違っています💢💢💢💢💢";
}
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8" name="robots" content="noindex">
	<title>研究室内限定(Local info) - ログイン</title>

	<!-- 共有ファイル -->
	<?php include($_SERVER['DOCUMENT_ROOT'].'/include.php'); ?>

	<!-- jQueryプログラム -->
	<script type="text/javascript" src="/local/local.js"></script>
</head>

<!-- ヘッダー埋め込み -->
<?php include(__DIR__.'/header.php'); ?>

<body>
	<div class="main">
		<h1>研究室内限定(Local info) - ログイン</h1>

		<?php
		if($error_message) {
			echo $error_message;
		}
		?>

		<form action="index.php" method="POST">
			<p>パスワード：<input type="password" name="password"></p>
			<input type="submit" name="login" value="ログイン">
		</form>
	</div>
</body>
</html>
