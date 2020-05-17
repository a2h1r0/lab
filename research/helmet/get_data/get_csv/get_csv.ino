#define Sensors 16  // Arduino1機に繋いでいるセンサの数

unsigned long base, now;  // 時間計測用変数
float voltage[Sensors];   // 電圧値用変数
int first = 0;  // 初回判別用変数
int i;          // ループカウンタ
 
void setup() {
  Serial.begin(57600);
}

void loop() {
  if (Serial.available() > 0) {   // PC側でser.writeが実行されれば真に
    if (first == 0) { // 取得開始時刻を保存
      base = micros();
      first++;
    }
    
    for (i=0; i<Sensors; i++) {
      voltage[i] = (analogRead(i)/1024.0)*5.0;  // 電圧の取得と5V化
      if(voltage[i] >= 4.9)       // 誤差の除去
        voltage[i] = 4.99;
      Serial.print(voltage[i]);   // 電圧値をPC側に送信
      Serial.print(" ");
    }
    now = micros() - base;  // 現在時刻と取得開始時刻の差を計算
    Serial.print(now);      // 取得時間をPC側に送信
    Serial.print("\n");     // 全てのセンサの電圧，時間を取得できたら改行

    Serial.read();  // Serial.available()を0に戻す
  }
}
