<!-- 認証処理 -->
<?php
session_start();

if(!isset($_SESSION["password"])) {
	header("Location: /iis-lab/index.php");
	exit;
}
?>
