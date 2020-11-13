// 16進数文字列のシリアル入力で色変更するサンプル

//Pick some colours, for example I like the look of #66FF99 and then convert this 24 bit 8:8:8 value to 5:6:5 or 6:6:6 using the most significant bits. So for this lovely greeney-blue the 66FF99 is:
//
//R: 0x66 = 0b01100110
//G: 0xFF = 0b11111111
//B: 0x99 = 0b10011001
//To convert to 18 bit (6:6:6) just take the top 6 bits of each:
//
//R: 011001
//G: 111111
//B: 100110
//then add them together to make the 18bits
//
//011001 111111 100110
//grouping that in fours:
//
//01 1001 1111 1110 0110
//which in hex is:
//
//0x19FE6
//Doing the same for 16 bit (5:6:5) would be:
//
//R: 01100
//G: 111111
//B: 10011
//which is
//
//0110 0111 1111 0011
//0x67F3
//So "limey green" is 0x19FE6 in 18 bit mode and 0x67F3 in 16 bit mode.
//
//Obviously you'd write a C program that you feed a 24 bit HTML #66FF99 from the tool and it would simply pump out the 16 and 18 bit values for it having done the and'ing and shifting for you rather than having to do it all manually.

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
char color[7] = {'\0'};

/**
 * @fn
 * シリアルから送信される表示色の受け取り
 * @param *color 表示色格納配列
 * @return void
 */
void receiveColor(char *color)
{
  int i = 0;
  char byte;

  while (1)
  {
    if (Serial.available())
    {
      byte = Serial.read();
      color[i] = byte;
      if (byte == '\n')
        break;
      i++;
    }
  }
  color[i] = '\0'; // \0: end of string
}

void setup()
{
  Serial.begin(9600);
  tft.begin();
}

void loop()
{
  receiveColor(color);
  Serial.println(color);
  tft.fillRect(150, 150, DRAW_SIZE, DRAW_SIZE, strtol(color, NULL, 16));
}
