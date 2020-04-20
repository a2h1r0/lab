<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>知的インタラクティブシステム研究室 - トップページ</title>

  <!-- 共有ファイル -->
  <?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/include.php'); ?>
  <!-- 個別CSS -->
  <link rel="stylesheet" type="text/css" href="/iis-lab/css/index.css">
</head>

<!-- ヘッダー埋め込み -->
<?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/parts/header.php'); ?>

<body>
  <!-- 全画面黒表示，動画だけ見せる -->
  <div class="top">
    <div class="top-video">
      <video autoplay muted>
        <source src="/iis-lab/resource/index.mp4" type="video/mp4" />
        <source src="/iis-lab/resource/index.webm" type="video/webm" />
        <p>知的インタラクティブシステム研究室(村尾研究室)</p>
      </video>

      <!-- 下矢印 -->
      <div class="downs">
        <a href="#scroll" class="down"></a>
      </div>
    </div>
  </div>


  <div class="main" id="scroll">
    <!-- About us -->
    <!-- タイトルだけ独立 -->
    <div>
      <h1>About Us??</h1>
    </div>

    <!-- 親エリアを定義 -->
    <section class="about">
      <!-- アニメーションを付けるため，要素のエリアを定義 -->
      <div class="detail">
        <p>知的インタラクティブシステム研究室は2017年4月に発足した研究室です．<br><br>ユビキタスコンピューティング，モバイルコンピューティング，ウェアラブルコンピューティングにおけるセンサ情報処理技術やインタフェースに関する研究を行っています．村尾准教授と双見助教授のご指導の下，メンバーは仲良く研究を進めており，学会や研究会にも毎年数回参加しています．また，長期休みの研究旅行の他，飲み会などのイベントも毎月(?)開催しています．明るい雰囲気で風通しの良い研究室です♪<br><br>村尾准教授の業績などは <a href="http://www.muraokazuya.net/"><i class="fa fa-caret-right"></i> muraokazuya.net</a> をご覧ください．</p>
      </div>
      <!-- アニメーションを付けるため，画像のエリアを定義 -->
      <div class="pic">
        <img src="/iis-lab/resource/index.jpg">
      </div>
    </section>
    <!-- ここまで -->


    <!-- ニュース -->
    <section class="news">
      <h1>Top News!!</h1>

      <ul>
        <!-- ニュースごとに<li>で囲む．ブログもあるので，最新10件ぐらいで良いと思う． -->
        <li>
          <time datetime="2019-09-18">2019/09/18</time>
          <p>大連理工大学からの3回編入生1名が配属されました．</p>
        </li>

        <li>
          <time datetime="2019-09-11">2019/09/11</time>
          <p>M1の吉田がロンドンで開催されたACM ISWC2019でBest Paperを受賞しました．</p>
        </li>

        <li>
          <time datetime="2019-07-11">2019/07/11</time>
          <p>3回生11名が配属されました．</p>
        </li>

        <li>
          <time datetime="2019-07-06">2018/04/01</time>
          <p>M1の梶原，澤野，吉田，西井が福島県で開催された情報処理学会DICOMO2019で発表しました．吉田，西井がヤングリサーチャー賞を受賞しました．研究室がナイトテクニカルセッションで3位入賞しました．</p>
        </li>

        <li>
          <time datetime="2018-07-11">2018/07/11</time>
          <p>3回生10名が配属されました．</p>
        </li>

        <li>
          <time datetime="2018-04-01">2018/04/01</time>
          <p>4月1日付で双見京介助教が着任しました．</p>
        </li>
      </ul>
    </section>
    <!-- ここまで -->


    <!-- メンバー -->
    <section class="member">
      <h1>Member</h1>

      <div class="professor">  <!-- 教授 -->
        <ul>
          <h1>Professor</h1>
          <li>准教授　村尾 和哉</li>
          <li>助教授　双見 京介</li>
          <li>研究補助員　松浪 里美</li>
        </ul>
      </div>


      <!-- それぞれ学年ごとに<ul>で囲む． -->
      <!-- 人数の多い学年はmember-left，member-rightで分ける -->

      <div class="doctor">  <!-- 博士 -->
        <ul>
          <h1>D3</h1>
          <li>NULL</li>
        </ul>

        <ul>
          <h1>D2</h1>
          <li>NULL</li>
        </ul>

        <ul>
          <h1>D1</h1>
          <li>NULL</li>
        </ul>
      </div>


      <div class="master">  <!-- 修士 -->
        <ul>
          <h1>M2</h1>
          <li>岡本 雅弘</li>
          <li>梶原 大暉</li>
          <li>澤野 亮太</li>
          <li>永松 悠大</li>
          <li>西井 遥菜</li>
          <li>吉田 航輝</li>
        </ul>

        <ul>
          <h1>M1</h1>
          <li>秋元 優摩</li>
          <li>八田 将志</li>
          <li>藤井 敦寛</li>
        </ul>
      </div>


      <div class="bachelor">  <!-- 学部 -->
        <ul>
          <h1>B4</h1>
          <div class="member-left">
            <li>磯部 海斗</li>
            <li>岡本 真梨菜</li>
            <li>奥田 大智</li>
            <li>小川 諒馬</li>
            <li>蔵田 直生</li>
            <li>斉藤 俊介</li>
          </div>
          <div class="member-right">
            <li>白井 希一</li>
            <li>田渕 裕貴</li>
            <li>平山 菜々華</li>
            <li>堀川 瑞生</li>
            <li>梁瀬 貞裕</li>
            <li>渡辺 大将</li>
            <li>陳 航</li>
          </div>
        </ul>

        <ul>
          <h1>B3</h1>
          <li>NULL</li>
        </ul>
      </div>
    </section>
    <!-- ここまで -->


    <!-- コース案内 -->
    <section class="course">
      <article>
        <h1>For B3 students</h1>
        <div class="button">
          <a href="/iis-lab/pages/introduction.html">学部生のためのページ</a>
          <a href="/iis-lab/pages/blog.html">研究室の様子</a>
        </div>
        <p>学部3回生のための研究室配属に関する情報は<a href="http://www.sa.ise.ritsumei.ac.jp/" class="univ">SAコースホームページ</a>において公開されます．</p>
      </article>

      <article>
        <h1>For B1 students</h1>
        <p>村尾研究室はシステムアーキテクトコース（SA）コース所属です．<br>将来的に村尾研究室への配属を希望する学生は，コース配属において<font color="red">SAコースに配属される必要</font>があります．<br>学部1回生のための研究室配属に関する情報は<a href="http://www.sa.ise.ritsumei.ac.jp/" class="univ">SAコースホームページ</a>において公開されます．</p>
      </article>
    </section>
    <!-- ここまで -->
  </div>
</body>

<!-- フッター埋め込み -->
<?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/parts/footer.php'); ?>
</html>
