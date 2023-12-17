import websockets.*;
import controlP5.*;

WebsocketServer ws;
int eyeMoveUpDown = 0;
int eyeMoveLeftRight = 0;
int blinkSpeed = 0;
int blinkStrength = 0;
ControlP5 cp5;
Chart myChart;

void setup() {
  //ws = new WebsocketServer(this, 19999, "/");

  size(400, 700);
  smooth();
  cp5 = new ControlP5(this);
  myChart = cp5.addChart("hello")
    .setPosition(50, 50)
    .setSize(200, 200)
    .setRange(-20, 20)
    .setView(Chart.LINE)
    ;

  myChart.getColor().setBackground(color(255, 100));


  myChart.addDataSet("world");
  myChart.setColors("world", color(255, 0, 255), color(255, 0, 0));
  myChart.setData("world", new float[4]);
}

void draw() {
  background(0);

  myChart.unshift("world", (sin(frameCount*0.01)*10));

  //textAlign(CENTER);
  //text("eyeMoveUpDown: " + eyeMoveUpDown, width/2, height/4);
  //text("eyeMoveLeftRight: " + eyeMoveLeftRight, width/2, height*2/4);
  //text("blinkSpeed: " + blinkSpeed, width/2, height*3/4);
  //text("blinkStrength: " + blinkStrength, width/2, height*4/4);
}

void webSocketServerEvent(String data) {
  int eyeMoveUp = int(split(split(data, "\"eyeMoveUp\":")[1], ",")[0]);
  int eyeMoveDown = int(split(split(data, "\"eyeMoveDown\":")[1], ",")[0]);
  eyeMoveUpDown = eyeMoveUp - eyeMoveDown;

  int eyeMoveLeft = int(split(split(data, "\"eyeMoveLeft\":")[1], ",")[0]);
  int eyeMoveRight = int(split(split(data, "\"eyeMoveRight\":")[1], ",")[0]);
  eyeMoveLeftRight = eyeMoveLeft - eyeMoveRight;

  blinkSpeed = int(split(split(data, "\"blinkSpeed\":")[1], ",")[0]);
  blinkStrength = int(split(split(data, "\"blinkStrength\":")[1], ",")[0]);
}
