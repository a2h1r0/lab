import websockets.*;

WebsocketServer ws;
String data = "";

void setup(){
  size(300, 150);
  ws = new WebsocketServer(this, 19999, "/");
}

void draw(){
  background(0);
  textAlign(CENTER);
  text(data, width, height);
}

void webSocketServerEvent(String msg){
  data = msg;
}
