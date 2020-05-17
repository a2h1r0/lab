#define Sensors 16  // Arduino1機に繋いでいるセンサの数

float voltage[Sensors];
int i;  // ループカウンタ
 
void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {   // PC側でser.writeが実行されれば真に
    for (i=0; i<Sensors; i++) {
      voltage[i] = (analogRead(i)/1024.0)*5.0;  // 電圧の取得と5V化
      if(voltage[i] >= 4.9)   // 誤差の除去
        voltage[i] = 4.99;
      Serial.print(voltage[i]);   // PC側に送信
      if(i < Sensors-1)
        Serial.print(" ");
    }
    Serial.print("\n");   // 全てのセンサの電圧を取得できたら改行
    
    Serial.read();  // Serial.available()を0に戻す
  }
}
