import websockets.*;

WebsocketServer ws;
int eyeMoveUpDown = 0;
int eyeMoveLeftRight = 0;
int blinkSpeed = 0;
int blinkStrength = 0;

void setup(){
  size(300, 150);
  ws = new WebsocketServer(this, 19999, "/");
}

void draw(){
  background(0);
  textAlign(CENTER);
  text("eyeMoveUpDown: " + eyeMoveUpDown, width/2, height/4);
  text("eyeMoveLeftRight: " + eyeMoveLeftRight, width/2, height*2/4);
  text("blinkSpeed: " + blinkSpeed, width/2, height*3/4);
  text("blinkStrength: " + blinkStrength, width/2, height*4/4);
}

void webSocketServerEvent(String data){
  int eyeMoveUp = int(split(split(data, "\"eyeMoveUp\":")[1], ",")[0]);
  int eyeMoveDown = int(split(split(data, "\"eyeMoveDown\":")[1], ",")[0]);
  eyeMoveUpDown = eyeMoveUp - eyeMoveDown;

  int eyeMoveLeft = int(split(split(data, "\"eyeMoveLeft\":")[1], ",")[0]);
  int eyeMoveRight = int(split(split(data, "\"eyeMoveRight\":")[1], ",")[0]);
  eyeMoveLeftRight = eyeMoveLeft - eyeMoveRight;

  blinkSpeed = int(split(split(data, "\"blinkSpeed\":")[1], ",")[0]);
  blinkStrength = int(split(split(data, "\"blinkStrength\":")[1], ",")[0]);
}
