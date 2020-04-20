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
    <h1>Hardware exercise</h1>

    <table>
      <tr>
        <td>
          <ul>
            <li>Arduinoでいろいろやってみる．</li>
          </ul>
          <ol>
            <ul>
              <li>Arduino工作入門（前編）<a href="http://deviceplus.jp/hobby/arduino-listicle-01/" rel="nofollow">リンク</a></li>
            </ul>
          </ol>
          <ul>
            <ul>
              <ul>
                <li>２－２までは超基本なので必ずやること．</li>
                <li>２－３，２－４は必要に応じてやる．</li>
                <ul>
                  <li>センサ，LED，抵抗，ブレッドボード，ワイヤ等は研究室にあります．もし不足しているもの（残り少ないもの）があれば購入しますので村尾まで知らせてください．</li>
                </ul>
                <li>Arduinoを使って以下のことをできるようにしておく．（上記２－２をやればできるはず）</li>
                <ul>
                  <li>スイッチ＋LED点灯</li>
                  <ul>
                    <li>Arduinoから電源供給して，スイッチを押したらLEDが光る回路をブレッドボード上で作る．</li>
                  </ul>
                  <li>ArduinoでLED点灯</li>
                  <ul>
                    <li>Arduinoにプログラムを書き込んで一定周期でLEDを点滅させる．</li>
                  </ul>
                  <li>照度センサ＋LED点灯</li>
                  <ul>
                    <li>光を当てたら（or暗くしたら）LEDが光るように変える．</li>
                  </ul>
                  <li>Arduino＋照度センサ＋LED点灯</li>
                  <ul>
                    <li>Arduinoにプログラムを書き込んで，照度センサの値をいったんArduinoが読み取って，光を当てたら（or暗くしたら）プログラム内で処理してLEDが光るようにする．</li>
                  </ul>
                </ul>
                <li><span>ArduinoとPCの通信もできるようにする</span></li>
                <ul>
                  <li>シリアルモニタ</li>
                  <ul>
                    <li>Arduino開発環境付属のシリアルモニタでセンサ値を表示させる</li>
                  </ul>
                  <li>シリアルプロッタ</li>
                  <ul>
                    <li>Arduino開発環境付属のシリアルプロッタでセンサの波形を表示させる</li>
                    <li><a href="https://qiita.com/umi_kappa/items/632c02d5d749004619ef" rel="nofollow">リンク</a></li>
                  </ul>
                  <li>外部ターミナルとArduinoの通信</li>
                  <ul>
                    <li>Tera Termという通信用ソフトウェアをつかってシリアルモニタと同様のことをする</li>
                    <ul>
                      <li><a href="https://ja.osdn.net/projects/ttssh2/" rel="nofollow">リンク</a></li>
                    </ul>
                    <li>パソコンにUSBやBluetooth接続された機器にはCOMポートという番号が付与される．その番号はシリアルポートと呼ばれていて，そのポートを介してシリアル通信を行う（シリアルモニタはそれを勝手にやってくれていた）</li>
                    <li>このあたりを見てやってみること</li>
                    <ul>
                      <li><a href="http://www.hiramine.com/physicalcomputing/raspberrypi/serial_howtoconnectpc.html" rel="nofollow">リンク</a></li>
                    </ul>
                    <li>余裕があれば2台のPCで研究室LAN内でTCP/IPの接続もやってみる．</li>
                  </ul>
                  <li>自作プログラムとArduinoの通信</li>
                  <ul>
                    <li>自分で書いたプログラムでCOMポートを経由してArduinoと通信できる．</li>
                    <li>すると，データを自在に処理して保存や描画できる．</li>
                    <li>どのプログラム言語でもできる．</li>
                    <li>Pythonだとこんな感じ（センサ→Arduino→PC→Python）</li>
                    <ul>
                      <li><a href="http://denshi.blog.jp/arduino/slope-python-graph" rel="nofollow">リンク</a></li>
                    </ul>
                    <li>PythonからArduinoにコマンドを送ってLEDを点灯させたりもできる</li>
                    <ul>
                      <li><a href="http://denshi.blog.jp/arduino/serial_led_python" rel="nofollow">リンク</a></li>
                    </ul>
                  </ul>
                </ul>
              </ul>
              <li>Arduino工作入門（後編）<a href="http://deviceplus.jp/hobby/arduino-listicle-02/" rel="nofollow">リンク</a></li>
              <ul>
                <li>後編は必要に応じてでよい．</li>
              </ul>
            </ul>
            <li>ユニバーサル基板で回路を作成してみる．</li>
            <ul>
              <li>Arduinoの回ではブレッドボード上で抵抗やLEDの回路を作っていたが，ブレッドボードはあくまでテストするもの．</li>
              <li>ユニバーサル基板にはんだ付けして回路を作成する．</li>
              <ul>
                <li>回路作成のやりかた <a href="http://www.murata.com/ja-jp/campaign/ads/japan/elekids/ele/craft/knack/universal" rel="nofollow">リンク</a><br>
                  <div style="display:block;text-align:left"><a href="P-03229.jpg" imageanchor="1"><img border="0" src="P-03229.jpg"></a></div>
                </li>
              </ul>
            </ul>
            <li>PCBで回路を作成してみる<b><font color="#ff0000">（これ以下の課題は今まで飛ばしたので，余裕があればやる程度）</font></b></li>
            <ul>
              <li>Arduinoシールドとしてデザインして外部発注し，部品を実装して作成する</li>
              <ul>
                <li>デザインするソフトウェア</li>
                <ul>
                  <li>Eagle <a href="https://www.autodesk.co.jp/products/eagle/overview" rel="nofollow">リンク</a></li>
                </ul>
                <li>デザインのデータを送ったら基板を作ってくれる会社</li>
                <ul>
                  <li>P板.COM <a href="https://www.p-ban.com/" rel="nofollow">リンク</a></li>
                  <li>Fusion PCB <a href="https://fusionpcb.jp/" rel="nofollow">リンク</a></li>
                </ul>
                <li>
                  <div style="display:block;text-align:left"><a href="20170607102936465.jpg" imageanchor="1"><img border="0" height="240" src="20170607102936465.jpg" width="320"></a></div><br>
                </li>
              </ul>
              <li>好きな回路をデザインして自分で作成する<br></li>
            </ul>
            <li>銅板をエッチングして回路を作成してみる．</li>
            <ul>
              <li>業者を使わずに自分で銅板に配線を印刷して，溶液で銅を溶かして回路を作成する．</li>
              <li>
                <div style="display:block;text-align:left"><a href="kadai01.jpg" imageanchor="1"><img border="0" src="kadai01.jpg"></a></div><br>
              </li>
            </ul>
          </ul>
        </td>
      </tr>
    </table>

  </div>
</body>
</html>
