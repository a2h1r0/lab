final int PORT = 10000;	// 通信に使用するポート番号
final int WINDOW_SIZE = 300; // ウィンドウサイズ


import processing.net.*;

Server server; // サーバ設定用変数

/* ウィンドウの初期化 */
void settings() {
	// ウィンドウサイズの設定
	size(WINDOW_SIZE, WINDOW_SIZE);
}

/* 初期化 */
void setup() {
    // サーバの設定
    server = new Server(this, PORT);
    println("Server address : " + server.ip());
    
    // フレームレートの設定
    frameRate(60);
}

/* メインループ */
void draw() {
	// クライアントの存在確認
	Client client = server.available();
	if (client != null) {
		// データの受信
		String received_data = client.readStringUntil('\0').trim();
		if (received_data != null) {
			// 背景色の変更
			background(int(received_data));

			// delayなしの状態で処理時間で10msほどかかる
			// println(millis());

			// 描画完了通知の送信
			client.write('0');
		}
	}
}