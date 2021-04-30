package com.example.getpulse;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import com.google.android.gms.common.api.ResultCallback;
import com.google.android.gms.wearable.MessageApi;
import com.google.android.gms.wearable.Wearable;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity implements SensorEventListener {

    private TextView mHeartView, mPulseView;
    private Button mRecordButton;
    private SensorManager mSensorManager;
    private int mTypeHeartRate, mTypeRawPPG;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        画面の常時点灯
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

//        変数定義
        mHeartView = (TextView) findViewById(R.id.heart);
        mPulseView = (TextView) findViewById(R.id.pulse);
        mRecordButton = findViewById(R.id.record);
//        センサ変数
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mTypeHeartRate = Sensor.TYPE_HEART_RATE;

//        センサ一覧の取得
        List<Sensor> sensorList = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        for (Sensor currentSensor : sensorList) {
            String name = currentSensor.getName();
            String type = currentSensor.getStringType();
            int number = currentSensor.getType();
            Log.d("Sensors", "Name: " + name + " / Type: " + type + " / Number: " + number);

//            PPGセンサのセンサ番号を取得
            if (type.contains("ppg")) {
                mTypeRawPPG = number;
                Log.d("PPG Sensor", "PPG Sensor: " + mTypeRawPPG);
            }
        }



        mRecordButton.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view){
//                ToDo: センサデータの保存処理
//            is_recording = true;
//            is_auto = true;
//            timestamps.clear(); //arraylistの要素を削除
//            values.clear();
//            //現在日時の取得
//            date = new Date();
//            startTime = System.currentTimeMillis();
//            //sensorValues = new StringBuilder();
//            Log.d("CSV", "Start Sensing");
            }
        });
    }


    @Override
    protected void onResume() {
        super.onResume();

//        センサの接続
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(mTypeHeartRate), SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(mTypeRawPPG), SensorManager.SENSOR_DELAY_FASTEST);
    }


    @Override
    protected void onPause() {
        super.onPause();

//        センサの接続終了
        mSensorManager.unregisterListener(this);
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
//        処理なし
    }


    @Override
    public void onSensorChanged(SensorEvent event) {
        float mHeartValue;
        int mPulseValue;

        Sensor sensor = event.sensor;
        if (sensor.getType() == Sensor.TYPE_HEART_RATE) {
//            心拍数センサ値の更新
            mHeartValue = event.values[0];
            mHeartView.setText(String.format("Heart: %f" , mHeartValue));
        } else if (sensor.getType() == 65572) {
//            脈波センサ値の更新
            mPulseValue = (int) event.values[0];
            mPulseView.setText(String.format("Pulse: %d" , mPulseValue));

//            ToDo: センサデータの保存処理
//            if(is_recording == true) {
//                //Log.d("CSV", Integer.toString(ppg));
//                long endTime = System.currentTimeMillis();
//                // カウント時間 = 経過時間 - 開始時間
//                long diffTime = (endTime - startTime);
//                //Log.d("CSV", Long.toString(diffTime));
//                SimpleDateFormat dataFormat = new SimpleDateFormat("mm:ss.SS", Locale.US);
//                tTextView.setText(dataFormat.format(diffTime));
//
//                timestamps.add((float)diffTime / 1000); //ArrayListに脈波データを追加
//                values.add(mPulseValue);
//
//                if(is_auto == true) {
//                    if(diffTime >= 30000) //何ミリ秒で計測終了するか
//                        endbutton.performClick(); //endbuttonを押す
//                }
//            }
        }
    }
}