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
    <h1>Software to be installed</h1>

    <table>
      <tr>
        <td>
          <ul>
            <li>Windows</li>
            <ul>
              <li>アップグレードする場合，大学のライセンスがあります．<b>（学内でインストールする必要あり）</b></li>
              <li>ただし，ドライバを入れるなどは自己責任．</li>
            </ul>
            <li>MS Office</li>
            <ul>
              <li>大学のライセンスがあります．<b>（学内でインストールする必要あり）</b></li>
              <ul>
                <li>【注意】学生の私物PCにインストールできる大学のOffice365というものがありますが，Office365は自分のお金で購入したPCのみへのインストールが認められています．<a href="https://secure.ritsumei.ac.jp/students/office365/column/detail/?category=Office%20ProPlus&amp;id=62" rel="nofollow">リンク</a><br>研究室で購入したPCは別のライセンスがあり，教員経由でのみインストール可能です．</li>
              </ul>
            </ul>
            <li>Adobe系</li>
            <ul>
              <li>大学のライセンスがあります．<b>（学内でインストールする必要あり）</b></li>
              <ul>
                <li>Acrobat（PDFの閲覧など）は必須．基本的な作業ならフリー版でOK．</li>
                <li>Photoshop（写真編集），Illustrator（イラスト作成）は使えたほうが良い．</li>
                <li>Premier（動画編集）もデモビデオ作成には必要．</li>
                <li>After Effects（動画編集）はPremiereには無い派手な編集が可能．</li>
              </ul>
            </ul>
            <li>ウィルス対策<b>（基本的に不要です）</b></li>
            <ul>
              <li>McAfeeが大学ライセンスであります．</li>
              <li>ただし，WindowsはWindows Defenderがもとからはいってるので新たに入れる必要はなし．</li>
            </ul>
            <li>TeX</li>
            <ul>
              <li>TeX（TeXインストーラ）<a href="http://www.math.sci.hokudai.ac.jp/~abenori/soft/bin/abtexinst_0_86.zip" rel="nofollow">リンク</a></li>
              <ul>
                <li>使い方はググるとたくさんでてきます．</li>
              </ul>
              <li>エディタ（Texmaker）<a href="http://www.xm1math.net/texmaker/assets/files/Texmaker_5.0.2_Win_x64.msi" rel="nofollow">リンク</a></li>
              <ul>
                <li>Texmakerの設定（下の図のようにする）</li>
                <div style="display:block;text-align:left">
                  <a href="tex.png" imageanchor="1"><img border="0" src="tex.png" style="max-width:100%"></a>
                </div>
                <ul>
                  <li>Commands</li>
                  <ul>
                    <li>LaTeX　platex -kanji=utf8 -src-specials -interaction=nonstopmode -jobname=% %.tex</li>
                    <li>PdfLaTeX　pdflatex -synctex=1 -interaction=nonstopmode %.tex</li>
                    <li>Bib(la)tex　pbibtex -kanji=utf8 %</li>
                    <li>dvips　"C:/w32tex/bin64/dvi2ps.exe" -o %.ps %.dvi</li>
                    <li>Dvipdfm　"C:/w32tex/bin64/dvipdfmx.exe" %.dvi</li>
                    <li>Dvi Viewer　"C:/w32tex/dviout/dviout.exe" %.dvi</li>
                  </ul>
                  <li>Quick Build</li>
                  <ul>
                    <li>Quick Build Command　LaTeX+dvipdfm+View PDF</li>
                  </ul>
                </ul>
              </ul>
            </ul>
            <li>プリンタドライバ</li>
            <ul>
              <li>MC780 のWindows10 64bit版をDL　<a href="http://www5.okidata.co.jp/JSHIS163.nsf/SearchView/F3A5BE4EBE16F64649257BF80020F83B?OpenDocument&amp;charset=Shift_JIS" rel="nofollow">リンク</a>　あとは，下の図ようにネットワーク接続のプリンタが自動的に見つかるはず．（ただし，研究室ネットワークにつなぐ必要あり）<br>
                <div style="display:block;margin:5px auto;text-align:center"><a href="OKI.png" imageanchor="1"><img border="0" height="273" width="320" src="OKI.png"></a></div></li>
              </ul>
              <li>メーラ</li>
              <ul>
                <li>私はThunderbird．Windows標準のOutlookを使ってる人はあまり見ない．</li>
              </ul>
              <li>ブラウザ</li>
              <ul>
                <li>Chrome, IE, Firefoxなどがありますが，何でもよいです．</li>
              </ul>
              <li>開発環境</li>
              <ul>
                <li>Arduino IDE</li>
                <ul>
                  <li>Arduinoという小さなコンピュータを開発するための環境です．言語はProcessingに似ていますが，Arduinoを制御するための独自の関数などがあります．実験２のボードコンピュータより簡単です．</li>
                  <li>例えばこのあたりで勉強できます．<a href="http://tyk-systems.com/arduinobasic/arduinobasic.html" rel="nofollow">リンク</a></li>
                  <li>下記リンクから「Arduino IDE」の「Windows Installer」をダウンロードする．</li>
                  <ul>
                    <li><a href="https://www.arduino.cc/en/Main/Software" rel="nofollow">リンク</a></li>
                  </ul>
                  <li>インストール方法はここを参照</li>
                  <ul>
                    <li><a href="https://will-ikusei.blogspot.jp/2016/02/arduinoide.html">リンク</a></li>
                  </ul>
                </ul>
                <li>Visual Studio（Windowsのアプリケーションを開発できます．ほかにもできますが）</li>
                <ul>
                  <li>Visual Studio Communityというのが無償で利用できます．通常の研究で使う分にはこれで十分．</li>
                  <li>Windowsのプログラム「.exe」はほぼすべてこれで作れます．</li>
                  <li>下記リンクからダウンロードして，インストールしてください．</li>
                  <li><a href="https://www.visualstudio.com/ja/downloads/?rr=https%3A%2F%2Fwww.microsoft.com%2Fja-jp%2Fdev%2Fdefault.aspx" rel="nofollow">リンク</a></li>
                  <li>インストール方法はググればたくさん出てきます．</li>
                  <ul>
                    <li>たとえば　<a href="http://www.softantenna.com/wp/tips/visual-studio-2017-install/" rel="nofollow">リンク</a></li>
                    <li>ワークロードと言って，何の言語で開発するかを選択して，必要なものだけインストールします．</li>
                    <li>必須</li>
                    <ul>
                      <li>.NETデスクトップ開発</li>
                      <li>C++によるデスクトップ開発</li>
                    </ul>
                    <li>オプション</li>
                    <ul>
                      <li>データサイエンスと分析のアプリケーション</li>
                      <li>Python開発</li>
                      <li>Unityでのゲーム開発</li>
                      <li>C++でのモバイル開発</li>
                    </ul>
                  </ul>
                </ul>
                <li>Anaconda</li>
                <ul>
                  <li>Pythonを管理する環境．</li>
                  <li>Pythonには2系と3系があり，微妙に異なる．2系で作ったコードは3系で動かないこともある．逆も然り．</li>
                  <li>下記リンクからダウンロードして，インストールする．一応研究室としてそろえたほうが良いので，3系（3.6 version）で．</li>
                  <li><a href="https://www.anaconda.com/download/" rel="nofollow">リンク</a></li>
                  <li>インストールはここを参照　<a href="http://pythondatascience.plavox.info/python%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB/python%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB-windows" rel="nofollow">リンク</a></li>
                  <li>Python自体の勉強はこのあたり．</li>
                  <ul>
                    <li><a href="http://www.tohoho-web.com/python/" rel="nofollow">リンク1</a></li>
                    <li><a href="https://www.pythonweb.jp/tutorial/" rel="nofollow">リンク2</a></li>
                    <li><a href="http://www.python-izm.com/" rel="nofollow">リンク3</a></li>
                  </ul>
                </ul>
              </ul>
            </ul>
            <ul>
              <li>通信系</li>
              <ul>
                <li>WinSCP，FFFTP</li>
                <ul>
                  <li>サーバ等とファイルをやり取りする．</li>
                  <li>ホームページのhtmlファイルをWebサーバに置く時とかに使う．</li>
                </ul>
                <li>putty</li>
                <ul>
                  <li>サーバと通信する</li>
                  <li>Webサーバのhtmlファイルを直接いじる場合とかに使う．</li>
                </ul>
                <li>port forwarder</li>
                <ul>
                  <li>ポートフォワーディングする</li>
                  <li>ルータの向こう側の特定のマシンに入る時とかに使う．</li>
                  <li>多分今は使わない．</li>
                </ul>
                <li>Tera term</li>
                <ul>
                  <li>COMポートや遠隔地のマシンと通信する</li>
                  <li>センサを接続するとCOMポートというのが割当てられて，その番号を指定して通信するので，センサ使う人は必須．</li>
                </ul>
                <li>VPN</li>
                <ul>
                  <li>RAINBOWのサイトに説明がある．学外からでも学内と同様の接続．</li>
                  <li>中国からでもフィルタリングされずにみれる．</li>
                </ul>
                <li>開発系</li>
                <ul>
                  <li>Processing</li>
                  <ul>
                    <li>図形描画が得意なプログラム．波形を表示させるだけとかならこれでもよい．</li>
                    <li>高度なことはできない．</li>
                  </ul>
                  <li>Android Studio（Androidアプリ開発）</li>
                  <ul>
                    <li>スマホのセンサデータを採取する人は必須．言語は基本はJava．</li>
                  </ul>
                  <li>Weka（機械学習）</li>
                  <ul>
                    <li>GUIで機械学習できる．</li>
                  </ul>
                  <li>123dDesign（3Dモデル作成）</li>
                  <ul>
                    <li>3Dプリンタで印刷するときとかのデータを作成するソフト．</li>
                  </ul>
                  <li>Eagle（回路設計）</li>
                  <ul>
                    <li>基盤設計するときの回路図を作成するソフト．出来たデータを中国に送れば基盤が郵送される．</li>
                  </ul>
                </ul>
                <li>便利系</li>
                <ul>
                  <li>解凍ツール（Lhaplusとか）</li>
                  <li>Skype</li>
                  <li>Dropbox</li>
                  <li>Googledrive</li>
                  <li>エディタ（メモ帳）</li>
                  <li>デスクトップキャプチャ（AGDRec）</li>
                  <li>辞書（Pdic)</li>
                </ul>
              </ul>
            </ul>
          </td>
        </tr>
      </table>

    </div>
  </body>
  </html>
