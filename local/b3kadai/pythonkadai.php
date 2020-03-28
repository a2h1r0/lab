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
    <h1>PythonKadai</h1>

    <table>
      <tr>
        <td>
          <ul>
            <li>入出力，グラフ表示</li>
            <ul>
              <li>CSVファイルの読み込み</li>
              <ul>
                <li>numpy, pandasなど</li>
                <li>文字列を含むCSVをpandasで読んでDataFrameをnumpy配列に変換する</li>
              </ul>
              <li>配列のファイルへの書き出し</li>
              <ul>
                <li>CSVやTSVなど任意の形式で出せるようにする</li>
                <li>1行目に行の説明をつける</li>
              </ul>
              <li>グラフ表示</li>
              <ul>
                <li>サブプロット，タイトル，軸などを自在につけられるようにする</li>
              </ul>
            </ul>
            <li><span>波形の前処理</span></li>
            <ul>
              <li>生の波形から移動平均，RMSを計算する</li>
              <li>波形を一定区間で切り出す</li>
              <li>スライディングウィンドウで波形を切り出す．ウィンドウ幅，ステップ幅を自在にできるようにする</li>
              <li>ウィンドウから平均，分散，最大，最小，ZeroCrossRateなどの特徴量を出してCSVで保存する</li>
            </ul>
            <li><span>分類・認識処理</span></li>
            <ul>
              <li>抽出した特徴量でK-meansクラスタリングをする</li>
              <li>抽出した特徴量でK-NNで認識処理する</li>
              <li>時系列波形のどこでもいいので2つを切り出して，DTW距離を計算する．</li>
            </ul>
            <li><span>評価</span></li>
            <ul>
              <li>全データを10分割しそのうち9つを学習データ，残り1つをテストデータとしてK-NNまたはDTWで精度を計算する</li>
              <li>10分割交差検証する</li>
              <li>ConfusionMatrixを作る</li>
              <li>TruePositive，FalsePositive，TrueNegative，FalseNegativeを計算し，Recall，Precision，F-measureを計算する</li>
            </ul>
            <li>発展</li>
            <ul>
              <li>DTWのワーピングパスにマスクをかける．Sakoe-Chibaバンド，Itakura Parallelogram　https://arxiv.org/pdf/0903.0041.pdf</li>
              <li>K-NNでK=1以外のときでやって多数決する</li>
              <li>X-means，RandomForest，SVMなど</li>
              <li>SAX，n-gram，BoW，SuffixTree(Suffix Array)</li>
              <li>Piece-wise Linear Approximation (Sliding Window and Bottom-up)</li>
            </ul>
          </ul>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
