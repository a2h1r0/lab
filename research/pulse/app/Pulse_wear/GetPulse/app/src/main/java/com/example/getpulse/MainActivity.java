package com.example.getpulse;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import static android.graphics.Typeface.BOLD;
import static android.graphics.Typeface.DEFAULT_BOLD;

public class MainActivity extends WearableActivity implements SensorEventListener{
    private final String TAG = MainActivity.class.getName();
    private SensorManager mSensorManager;
    public float hb=100.0f;
    private TextView textView;
    private TextView heartTextView;
    public View backGround;
    public boolean isDisp=true;
    private LoopEngine loopEngine = new LoopEngine();
    public ImageView imageView;
    boolean set = false;
    @Override
    protected void onStart(){
        super.onStart();
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setAmbientEnabled();
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        textView = (TextView) findViewById(R.id.text);
        heartTextView = (TextView) findViewById(R.id.text_heart);
        backGround = (View) findViewById(R.id.View);
        textView.setTextSize(20.0f);
        heartTextView.setTextSize(0.0f);
        loopEngine.start();
        //heartTextView.setTextColor(Color.argb(80, 67, 135, 233));
        textView.setTextColor(Color.argb(255, 140, 140, 140));
        textView.setTypeface(Typeface.create(DEFAULT_BOLD, BOLD));
    }
    private LinearLayout.LayoutParams createParam(int w, int h){
        return new LinearLayout.LayoutParams(w, h);
    }
    @Override
    protected void onResume() {
        super.onResume();
        Sensor sensor = mSensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE);
        mSensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL);
    }
    @Override
    protected void onPause(){
        super.onPause();
        mSensorManager.unregisterListener(this);
    }
    @Override
    public void onSensorChanged(SensorEvent event) {
        //ここで変数宣言すると，起動中は破棄されずメモリリークする
        if(set==false)textView.setTextSize(60.0f);
        if (event.sensor.getType() == Sensor.TYPE_HEART_RATE) {
            hb = event.values[0];
            textView.setText(""+(int)hb);
            set = true;
        }
    }
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.d(TAG,"onAccuracyChanged!!");
    }
    public void update(){
        if(set) {
            if (isDisp) {
                backGround.setBackgroundColor(Color.argb(80, 231, 232, 226));
                //heartTextView.setTextSize(100.0f);
                textView.setTextSize(60.0f);
//                ImageView img = (ImageView) findViewById(R.id.imageView);
                Resources res = getResources();
//                Bitmap bitmap = BitmapFactory.decodeResource(res, R.drawable.heart);
                // bitmapの画像を250*250で作成する
//                Bitmap bitmap2 = Bitmap.createScaledBitmap(bitmap, 250, 250, false);
//                img.setImageBitmap(bitmap2);
            } else {
                backGround.setBackgroundColor(Color.argb(10, 231, 232, 226));
                //heartTextView.setTextSize(800.0f);
                textView.setTextSize(70.0f);
//                ImageView img = (ImageView) findViewById(R.id.imageView);
                Resources res = getResources();
//                Bitmap bitmap = BitmapFactory.decodeResource(res, R.drawable.heart);
                // bitmapの画像を300*300で作成する
//                Bitmap bitmap2 = Bitmap.createScaledBitmap(bitmap, 300, 300, false);
//                img.setImageBitmap(bitmap2);
            }
        }
        isDisp = !isDisp;
    }
    //一定時間後にupdateを呼ぶためのオブジェクト
    class LoopEngine extends Handler {
        private boolean isUpdate;
        public void start(){
            this.isUpdate = true;
            handleMessage(new Message());
        }
        public void stop(){
            this.isUpdate = false;
        }
        @Override
        public void handleMessage(Message msg) {
            this.removeMessages(0);//既存のメッセージは削除
            if(this.isUpdate){
                MainActivity.this.update();//自信が発したメッセージを取得してupdateを実行
                sendMessageDelayed(obtainMessage(0), (long)(60/hb*1000));//鼓動の間隔でメッセージを出力
            }
        }
    };
}