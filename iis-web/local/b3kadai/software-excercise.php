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
    <h1>Software exercise</h1>

    <table>
      <tr>
        <td>
          <div>以下の課題１～５を各自行う．課題6，7，8は今までやっていないので飛ばしてOK．</div>
          <ol>
            <li>Arduino＋圧力センサ＋コンソール出力<br>Arduinoで圧力センサの値を読み取り，USB接続したパソコン上のコンソール（ArduinoIDEでよい）に数値を表示する．</li>
            <li>Arduino＋任意のセンサ＋コンソール出力<br>1の内容を任意のセンサで行う．任意のセンサとは温度，湿度，気圧，カメラ，マイク，GPS，心電，脈拍などなんでもよい．</li>
            <li>Arduino＋任意のセンサ＋グラフ描画<br>Arduinoエディタのシリアルプロッタを使ってセンサ値をリアルタイムでグラフ表示する．</li>
            <li>Arduino＋センサ＋PCファイル保存<br>1および2でコンソール出力した値をファイルに保存する．</li>
            <li>コンソールからのコマンドでArduinoを制御（その１）<br>コンソールからArduinoに'0'を送信すると，Arduinoに接続されたLEDが消え，'1'を送信するとLEDがつくようにする．</li>
            <li>コンソールからのコマンドでArduinoを制御（その２）<br>コンソールからArduinoに'0'を送信すると，ArduinoからPCへのデータ送信を停止し，'1'を送信するとデータ送信を再開するようにする．</li>
            <li>（発展）BluetoothでPCにデータ送信</li>
            <li>（超発展）LAN経由で研究室内のマシンにデータ保存．</li>
          </ol>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
