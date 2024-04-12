import websockets.*;
import controlP5.*;

WebsocketServer ws;
ControlP5 cp5;
Chart blinkChart, eyeMoveChart;

int LENGTH = 1000;
int blinkSpeed, blinkStrength, eyeMoveUpDown, eyeMoveLeftRight = 0;

void setup() {
    size(1600, 950);
    
    ws = new WebsocketServer(this, 19999, "/");
    cp5 = new ControlP5(this);
    
    blinkChart = cp5.addChart("blinkChart")
       .setPosition(50, 500)
       .setSize(1500, 400)
       .setRange(0, 200)
       .setView(Chart.LINE);
    blinkChart.getColor().setBackground(#808080);
    
    blinkChart.addDataSet("blinkSpeed");
    blinkChart.setColors("blinkSpeed", #ff0000);
    blinkChart.setData("blinkSpeed", new float[LENGTH]);
    
    blinkChart.addDataSet("blinkStrength");
    blinkChart.setColors("blinkStrength", #0000ff);
    blinkChart.setData("blinkStrength", new float[LENGTH]);
    
    eyeMoveChart = cp5.addChart("eyeMoveChart")
       .setPosition(50, 50)
       .setSize(1500, 400)
       .setRange( -8, 8)
       .setView(Chart.LINE);    
    eyeMoveChart.getColor().setBackground(#808080);
    
    eyeMoveChart.addDataSet("eyeMoveUpDown");
    eyeMoveChart.setColors("eyeMoveUpDown", #ff0000);
    eyeMoveChart.setData("eyeMoveUpDown", new float[LENGTH]);
    
    eyeMoveChart.addDataSet("eyeMoveLeftRight");
    eyeMoveChart.setColors("eyeMoveLeftRight", #0000ff);
    eyeMoveChart.setData("eyeMoveLeftRight", new float[LENGTH]);    
}

void draw() {
    background(0);
    
    blinkChart.push("blinkSpeed", blinkSpeed);
    blinkChart.push("blinkStrength", blinkStrength);
    
    eyeMoveChart.push("eyeMoveUpDown", eyeMoveUpDown);
    eyeMoveChart.push("eyeMoveLeftRight", eyeMoveLeftRight);
}

void webSocketServerEvent(String data) {
    blinkSpeed = int(split(split(data, "\"blinkSpeed\":")[1], ",")[0]);
    blinkStrength = int(split(split(data, "\"blinkStrength\":")[1], ",")[0]);
    
    int eyeMoveUp = int(split(split(data, "\"eyeMoveUp\":")[1], ",")[0]);
    int eyeMoveDown = int(split(split(data, "\"eyeMoveDown\":")[1], ",")[0]);
    eyeMoveUpDown = eyeMoveUp - eyeMoveDown;
    
    int eyeMoveLeft = int(split(split(data, "\"eyeMoveLeft\":")[1], ",")[0]);
    int eyeMoveRight = int(split(split(data, "\"eyeMoveRight\":")[1], ",")[0]);
    eyeMoveLeftRight = eyeMoveLeft - eyeMoveRight;    
}
