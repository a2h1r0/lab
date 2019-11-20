#define Sensors 16  // Arduino1機に繋いでいるセンサの数

unsigned long time;
float voltage[Sensors];
int i;  // ループカウンタ
 
void setup() {
  Serial.begin(57600);
}

void loop() {
  if (Serial.available() > 0) {   // PC側でser.writeが実行されれば真に
    for (i=0; i<Sensors; i++) {
      voltage[i] = (analogRead(i)/1024.0)*5.0;  // 電圧の取得と5V化
      if(voltage[i] >= 4.9)   // 誤差の除去
        voltage[i] = 4.99;
      Serial.print(voltage[i]);   // PC側に送信
      Serial.print(" ");
    }
    time = micros();  // 動作開始からのマイクロ時間を取得
    Serial.print(time); // PC側に送信
    Serial.print("\n");   // 全てのセンサの電圧，時間を取得できたら改行

    Serial.read();  // Serial.available()を0に戻す
  }
}
