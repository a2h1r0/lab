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
    <h1>B3課題</h1>

    <table>
      <tr>
        <td>
          <ul>
            <li><a href="todofirst.php">配属が決まったらすぐにやる事項（基本的に初回顔合せ時に実施）</a></li>
            <li>
              <span><a href="pcsetup.php">PCのセットアップ（PC貸与後，8月上旬までに完了が望ましい）</a></span>
            </li>
            <li>
              <span>ミーティングおよび輪講</span>
            </li>
            <ul>
              <li>配属決定後，試験期間終了までは，授業が無ければ自由参加です．</li>
            </ul>
            <li>
              <a href="paperreading.php">論文課題</a>
            </li>
            <ul>
              <li>配属後，初回顔合せまたは2回目に課題を提示し，試験期間終了後から取り組んでもらう．</li>
              <li>8月第2週　【プレゼン】UWW(3本)の発表</li>
              <li>8月第3週　【プレゼン】DICOMO(1本)の発表</li>
              <li>8月第4週　【プレゼン】英語論文(1本)の発表</li>
              <li>発表する日などのスケジューリングは輪講担当が行う．発表は基本的にB4以上も全員参加．</li>
            </ul>
            <li>
              <a href="hardware-excercise.php">ハードウェア課題</a>
            </li>
            <ul>
              <li>9月第1週　【演習】Arduino+ブレッドボード，9月第4週にやるセンサの選択</li>
              <li>9月第2週　【演習】Arduino+ユニバーサル基盤（はんだ）</li>
              <li>9月第3週　【演習】PCBデザインと発注</li>
              <li>9月第4週　【演習】3Dプリンタ</li>
              <li>9月第4週　【プレゼン】選択したセンサのデータをコンソール等に表示</li>
            </ul>
            <li>
              <a href="software-excercise.php">ソフトウェア課題</a>
            </li>
            <ul>
              <li>10月第1週　ArduinoのデータをPythonで受信，ファイル保存（csvやpickle）</li>
              <li>10月第2週　ArduinoのデータをPythonで受信，Pyplotでグラフ描画（subplot, リアルタイム等様々なもの）</li>
            </ul>
            <li>
              <a href="data-processing-excercise.php">データ解析課題</a>
            </li>
            <ul>
              <li>10月第3週　CSVの加速度ファイルを読み込んでスライディングウィンドウで特徴量抽出（時間空間特徴のみ）</li>
              <li>10月第4週　FFT</li>
              <li>11月第1週　K-NN，DTW</li>
              <li>11月第2週　Scikit-learnでSVM使って分類（データは何でもよい）</li>
              <li>11月第3週　ChainerでDNN，CNN（１）</li>
              <li>11月第4週　ChainerでDNN，CNN（２）</li>
            </ul>
            <li>
              <a href="pythonkadai.php">Python課題</a>
            </li>
            <ul>
              <li>12月第1週　Cross validation，Confusion matrix，Recall，Precision，F-measure</li>
              <li>12月第2週　3Dプリンタ</li>
            </ul>
          </ul>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
