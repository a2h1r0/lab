#define Sensors 16  // Arduino1機に繋いでいるセンサの数
#define Name_SIZE 16

#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27,16,2);

float voltage[Sensors];   // 電圧値用変数
int i;          // ループカウンタ
char input;
String tester;
 
void setup() {
  Serial.begin(57600);
  lcd.init(); 
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Let's start!!");
}

void loop() {
  if (Serial.available() > 0) {   // PC側でser.writeが実行されれば真に
  input = Serial.read();

    if (input == '0') {
      lcd.clear();
      lcd.print("Who are you?");
    }

    else if (input == '1') {      
      for (i=0; i<Sensors; i++) {
        voltage[i] = (analogRead(i)/1024.0)*5.0;  // 電圧の取得と5V化
        if(voltage[i] >= 4.9)       // 誤差の除去
          voltage[i] = 4.99;
        Serial.print(voltage[i]);   // 電圧値をPC側に送信
        Serial.print(" ");
      }
      Serial.print("\n");     // 全てのセンサの電圧，時間を取得できたら改行
    }

  
    else if (input == '2') {
      lcd.clear();
      lcd.print("You are...");
      
      tester = Serial.readStringUntil('\0');
      lcd.setCursor(0, 1);
      lcd.print(tester);
    }

    else if (input == '3') {
      lcd.clear();
      lcd.print("Thank you");
      lcd.setCursor(3, 1);
      lcd.print("for watching!");
    }
  }
}
