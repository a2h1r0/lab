import websockets.*;

WebsocketServer ws;
PrintWriter file;

int blinkSpeed, blinkStrength, eyeMoveUpDown, eyeMoveLeftRight = 0;

void setup() {
    size(1600, 950);
    
    ws = new WebsocketServer(this, 19999, "/");
    file = createWriter("data.csv");
}

void draw() {
    background(0);
    
    file.print(blinkSpeed);
    file.print(",");
    file.print(blinkStrength);
    file.print(",");
    file.print(eyeMoveUpDown);
    file.print(",");
    file.println(eyeMoveLeftRight);
    
    // todo: 秒数カウントで切るなど
    if (x >= 100) {
        file.flush();
        file.close();
        exit();
    }
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
