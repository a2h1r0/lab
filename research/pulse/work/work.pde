final int PORT = 10000;	// 通信に使用するポート番号
final int WINDOW_SIZE = 500; // ウィンドウサイズ


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
	frameRate(120);
}

/* メインループ */
void draw() {
	// クライアントの存在確認
	Client client = server.available();
	if (client != null) {
		// データの受信
		String received_data = client.readStringUntil('\0').trim();
		if (received_data != null) {
			// 受信データの整形
			String[] data = split(received_data, ',');
			
			// 16進数化（色コード形式）
			String color_code = hex(int(data[1]), 6);
			// RGB化（2桁ずつ10進数化）
			int r = unhex(color_code.substring(0, 2));
			int g = unhex(color_code.substring(2, 4));
			int b = unhex(color_code.substring(4, 6));
			
			// 背景色の変更
			background(r, g, b);
			
			// 点灯時間の待機
			// delay(int(data[0]));
			
			// delayなしの状態で処理時間で10msほどかかる
			// println(millis());
			
			// 描画完了通知の送信
			client.write('0');
		}
	}
}