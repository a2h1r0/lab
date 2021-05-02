#include "Arduino.h"
#include "Adafruit_ILI9341.h"

//! ディスプレイのピン配置
#define TFT_DC 9
#define TFT_CS 10
#define TFT_RST 8
#define TFT_MISO 12
#define TFT_MOSI 11
#define TFT_CLK 13

//! ディスプレイクラス
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_MOSI, TFT_CLK, TFT_RST, TFT_MISO);

//! 描画サイズ
#define DRAW_SIZE 50

//! 表示色格納配列
char color[5] = {'\0'};

/**
 * @fn
 * 初期化
 */
void setup()
{
  Serial.begin(9600);
  tft.begin();
}

/**
 * @fn
 * メインループ
 */
void loop()
{
  // PCからデータが送信されると実行
  if (Serial.available() > 0)
  {
    // 表示色の受け取り（colorに格納）
    Serial.readStringUntil('\0').toCharArray(color, sizeof color);

    // 描画（atoiではオーバーフローする）
    tft.fillRect(10, 10, DRAW_SIZE, DRAW_SIZE, atol(color));

    // 描画の待機
    // delay(10);

    // 脈波の読み取り
    Serial.println(analogRead(A0));
  }
}
