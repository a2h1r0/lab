import processing.net.*;

int port = 10001; // 適当なポート番号を設定

Server server;

void setup() {
	// 画面サイズ
	size(300, 300);
	server = new Server(this, port);
	println("server address : " + server.ip()); // IPアドレスを出力
}

void draw() {
	Client client = server.available();
	if (client != null) {
		String whatClientSaid = client.readString();
		if (whatClientSaid != null) {
			background(int(whatClientSaid));
		}
	} 
}