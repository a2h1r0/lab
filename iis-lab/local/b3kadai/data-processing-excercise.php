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
    <h1>Data processing excercise</h1>

    <table>
      <tr>
        <td>
          <ol>
            <li>Wekaをつかってクラスタリング<br><a href="http://www.cs.waikato.ac.nz/ml/weka/downloading.html" rel="nofollow">ここ</a>からwekaをダウンロード．<br><a href="https://archive.ics.uci.edu/ml/datasets/Iris" rel="nofollow">ここ</a>からIrisのデータセットをダウンロード．<br>wekaを使って，Irisデータをクラスタリングする．<br>クラスタリングした結果とIrisの種類を比較する．Cluster Precisionを使うとよい．</li>
            <li>Wekaをつかってデータ分類<br>1ではクラスタリングをしたが，2ではIrisデータの一部を学習して，残りのIrisの種類を推定する．<br>クラスタリングと分類の違いを理解すること．<br>学習方法は10 fold cross validation<br>識別器はSVM (SMO)とRandomForestで行い，比較する．</li>
            <li>K-NN (K-Nearest Neighbor)を実装する．<br>K-NNを自分で実装し，Irisデータを分類する．<br><a href="http://qiita.com/yshi12/items/26771139672d40a0be32" rel="nofollow">参考ページ１</a></li>
            <li>ユークリッド距離での波形分類を実装する．<br><a href="http://mathtrain.jp/manhattan" rel="nofollow">参考ページ１</a><br></li>
            <li>DTW (Dynamic Time Warping)を実装する．<br><a href="http://centraleden.hatenablog.com/entry/2014/05/12/DTW%28dynamic_time_warping%29%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E6%99%82%E7%B3%BB%E5%88%97%E3%83%87%E3%83%BC%E3%82%BF%E3%82%92%E5%88%86%E9%A1%9E%E3%81%97%E3%81%A6%E3%81%BF%E3%81%9F" rel="nofollow">参考ページ１</a><br><a href="http://sinhrks.hatenablog.com/entry/2014/11/14/232603" rel="nofollow">参考ページ２</a></li>
          </ol>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
