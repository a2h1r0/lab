<!-- 認証処理 -->
<?php
session_start();

if(!isset($_SESSION["password"])) {
	header("Location: index.php");
	exit;
}
?>
