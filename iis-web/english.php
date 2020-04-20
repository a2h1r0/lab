<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Intelligent Interactive System Laboratory - Top</title>

  <!-- 共有ファイル -->
  <?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/include.php'); ?>
  <!-- 個別CSS -->
  <link rel="stylesheet" type="text/css" href="/iis-lab/css/index.css">
</head>

<!-- ヘッダー埋め込み -->
<?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/parts/en-header.php'); ?>

<body>
  <!-- 全画面黒表示，動画だけ見せてる -->
  <div class="top">
    <div class="top-video">
      <video autoplay muted>
        <source src="/iis-lab/resource/index.mp4" type="video/mp4" />
        <source src="/iis-lab/resource/index.webm" type="video/webm" />
        <p>Intelligent Interactive System Laboratory(Murao Lab.)</p>
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
        <p>Intelligent Interactive System Laboratory has been launched in April 2017.<br><br>We are working on sensor data processing and interfaces in the fields of ubiquitous, mobile, and wearable computing.<br><br>Regarding Kazuya MURAO's publications, please visit <a href="http://www.muraokazuya.net/"><i class="fa fa-caret-right"></i> muraokazuya.net</a></p>
      </div>
      <!-- アニメーションを付けるため，画像のエリアを定義 -->
      <div class="pic">
        <img src="/iis-lab/resource/introduction.jpg">
      </div>
    </section>
    <!-- ここまで -->


    <!-- ニュース -->
    <section class="news">
      <h1>Top News!!</h1>

      <ul>
        <!-- ニュースごとに<li>で囲む．ブログもあるので，最新10件ぐらいで良いと思う． -->
        <li>
          <time datetime="2019-09-18">Sep. 18, 2019</time>
          <p>One B3 student admitted from Dalian University of Technology joined our group.</p>
        </li>

        <li>
          <time datetime="2019-09-11">Sep. 11, 2019</time>
          <p>Kazuki Yoshida won the ACM ISWC2019 Best Paper.</p>
        </li>

        <li>
          <time datetime="2019-07-11">July 11, 2019</time>
          <p>Eleven B3 students joined our group.</p>
        </li>

        <li>
          <time datetime="2019-07-06">July 6, 2019</time>
          <p>Kajiwara, Sawano, Yoshida, and Nishii presented their works at IPSJ DICOMO2019. Yoshida and Nishii received the Young Researcher Award, and Murao lab got 3rd place at Night Technical Session.</p>
        </li>

        <li>
          <time datetime="2018-07-11">July 11, 2018</time>
          <p>Ten B3 students joined our group.</p>
        </li>

        <li>
          <time datetime="2018-04-01">April 1, 2018</time>
          <p>Dr. Kyosuke FUTAMI joined our group as an assistant professor as of Apr 1st.</p>
        </li>
      </ul>
    </section>
    <!-- ここまで -->


    <!-- メンバー -->
    <section>
      <h1>Member</h1>

      <div class="professor">  <!-- 教授 -->
        <ul>
          <h1>Professor</h1>
          <li>Associate Professor  Kazuya MURAO</li>
          <li>Assistant Professor  Kyosuke FUTAMI</li>
          <li>Assistant  Satomi MATSUNAMI</li>
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
          <li>Masahiro OKAMOTO</li>
          <li>Daiki KAJIWARA</li>
          <li>Ryota SAWANO</li>
          <li>Yuuki NAGAMATSU</li>
          <li>Haruna NISHII</li>
          <li>Kazuki YOSHIDA</li>
        </ul>

        <ul>
          <h1>M1</h1>
          <li>Yuma AKIMOTO</li>
          <li>Masashi HATTA</li>
          <li>Atsuhiro FUJII</li>
        </ul>
      </div>


      <div class="bachelor">  <!-- 学部 -->
        <ul>
          <h1>B4</h1>
          <div class="member-left">
            <li>Kaito ISOBE</li>
            <li>Marina OKAMOTO</li>
            <li>Daichi OKUDA</li>
            <li>Ryoma OGAWA</li>
            <li>Naoki KURATA</li>
            <li>Shunsuke SAITO</li>
          </div>
          <div class="member-right">
            <li>Kiichi SHIRAI</li>
            <li>Yuki TABUCHI</li>
            <li>Nanaka HIRAYAMA</li>
            <li>Mizuki HORIKAWA</li>
            <li>Sadahiro YANASE</li>
            <li>Daisuke WATANABE</li>
            <li>Hang Chen</li>
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
        <h1>Lab assignment info for B3 students</h1>
        <div class="button">
          <a href="/iis-lab/pages/introduction.html">学部生のためのページ</a>
          <a href="/iis-lab/pages/blog.html">研究室の様子</a>
        </div>
        <p>学部3回生のための研究室配属に関する情報は<a href="http://www.sa.ise.ritsumei.ac.jp/" class="univ">SAコースホームページ</a>において公開されます．</p>
      </article>

      <article>
        <h1>Course assignment info for B1 students</h1>
        <p>村尾研究室はシステムアーキテクトコース（SA）コース所属です．<br>将来的に村尾研究室への配属を希望する学生は，コース配属において<font color="red">SAコースに配属される必要</font>があります．<br>学部1回生のための研究室配属に関する情報は<a href="http://www.sa.ise.ritsumei.ac.jp/" class="univ">SAコースホームページ</a>において公開されます．</p>
      </article>
    </section>
    <!-- ここまで -->
  </div>
</body>

<!-- フッター埋め込み -->
<?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/parts/en-footer.php'); ?>
</html>
