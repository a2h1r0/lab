<!doctype html>
<html>

<head>
	<meta charset="utf-8">
	<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha256-4+XzXVhsDmqanXGHaHvgh1gMQKX40OUvDEBTu8JcmNs=" crossorigin="anonymous"></script>
</head>

<body>
	<style>
		body {
			width: 100%;
			height: 100vh;
		}
	</style>

	<script>
		// 黒で初期化
		$('body').css('background-color', 'rgb(140, 140, 140)');

		setInterval(() => {
			if ($('body').css('background-color') === 'rgb(140, 140, 140)') {
				// 黒の場合，白に変更
				$('body').css('background-color', 'rgb(160, 160, 160)');
			} else {
				// 白の場合，黒に変更
				$('body').css('background-color', 'rgb(140, 140, 140)');
			}
		}, 800);
	</script>
</body>

</html>