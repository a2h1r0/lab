package com.example.getpulse;

import android.app.Activity;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.common.api.ResultCallback;
import com.google.android.gms.wearable.MessageApi;
import com.google.android.gms.wearable.Wearable;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity implements SensorEventListener {

//    センサ値取得前の待機時間（キャリブレーション）
    private long mInitializationTime = 60;
//    センサ値取得時間
    private long mSensingTime = 60;

    private TextView mHeartView, mPulseView;
    private Button mRecordButton;
    private SensorManager mSensorManager;
    private int mTypeHeartRate, mTypeRawPPG;
    private boolean mIsRecording;
    private Date mRecordingDateTime;
    private long mStartRecordingTime;
    private float mFinishRecordingTime;
    private ArrayList<Long> mHeartTimestamps = new ArrayList<>(), mPulseTimestamps = new ArrayList<>();
    private ArrayList<Integer> mHeartValues = new ArrayList<>(), mPulseValues = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        画面の常時点灯
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

//        時間のミリ秒変換
        mInitializationTime *= 1000;
        mSensingTime *= 1000;
        mFinishRecordingTime = mInitializationTime + mSensingTime;

//        変数定義
        mHeartView = findViewById(R.id.heart);
        mPulseView = findViewById(R.id.pulse);
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


        /**
         * Recordボタンの押下で記録を開始
         */
        mRecordButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
//                ボタンの無効化
                mRecordButton.setEnabled(false);

//                初期化
                mHeartTimestamps.clear();
                mHeartValues.clear();
                mPulseTimestamps.clear();
                mPulseValues.clear();

//                取得開始時間の取得
                mRecordingDateTime = new Date();
                mStartRecordingTime = System.currentTimeMillis();

//                データの取得開始
                mIsRecording = true;
            }
        });
    }


    /**
     * 起動時にセンサを接続
     */
    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(mTypeHeartRate), SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(mTypeRawPPG), SensorManager.SENSOR_DELAY_FASTEST);
    }


    /**
     * 終了時にセンサの接続を解除
     */
    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }


    /**
     * センサの精度が更新された場合に実行
     * @param sensor 更新されたセンサ
     * @param accuracy 更新後の精度
     */
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
//        処理なし
    }


    /**
     * センサの値が更新された場合に実行
     * @param event センサの更新イベント
     */
    @Override
    public void onSensorChanged(SensorEvent event) {
        int heartValue = -1, pulseValue = -1;
        Sensor sensor = event.sensor;

        if (sensor.getType() == mTypeHeartRate) {
//            心拍数センサ値の更新
            heartValue = (int) event.values[0];
            mHeartView.setText(String.format("Heart: %d" , heartValue));
        } else if (sensor.getType() == mTypeRawPPG) {
//            脈波センサ値の更新
            pulseValue = (int) event.values[0];
            mPulseView.setText(String.format("Pulse: %d" , pulseValue));
        }

        if (mIsRecording) {
            long processingTime = System.currentTimeMillis() - mStartRecordingTime;

            if (processingTime < mInitializationTime) {
//                キャリブレーション中
                mRecordButton.setText(String.format("INITIALIZING...  %.2f", (float) (mInitializationTime - processingTime) / 1000));
                mRecordButton.setTextColor(Color.WHITE);
            } else if (mInitializationTime < processingTime && processingTime < mFinishRecordingTime) {
//                データ取得中
                mRecordButton.setText(String.format("RECORDING...  %.2f", (float) (mFinishRecordingTime - processingTime) / 1000));
                mRecordButton.setTextColor(Color.RED);

//                値が存在する場合は追加
                if (heartValue != -1) {
                    mHeartTimestamps.add(processingTime);
                    mHeartValues.add(heartValue);
                }
                if (pulseValue != -1) {
                    mPulseTimestamps.add(processingTime);
                    mPulseValues.add(pulseValue);
                }
            } else if (mFinishRecordingTime < processingTime) {
//                データの取得終了
                mIsRecording = false;

//                ファイルに保存
                try {
                    SaveToExternalStorage("HeartRate", mHeartTimestamps, mHeartValues);
                    SaveToExternalStorage("Pulse", mPulseTimestamps, mPulseValues);
                } catch (IOException e) {
                    Toast.makeText(getApplicationContext(), e.toString(), Toast.LENGTH_LONG).show();
                }

                Toast.makeText(getApplicationContext(), "データを保存しました！", Toast.LENGTH_LONG).show();
//                ボタンの有効化
                mRecordButton.setText("RECORD");
                mRecordButton.setTextColor(Color.WHITE);
                mRecordButton.setEnabled(true);
            }
        }
    }


    /**
     * データの保存処理
     * @param dataType データの種類（心拍数 or 脈波）
     * @param timestamps タイムスタンプ配列
     * @param values 値の配列
     */
    private void SaveToExternalStorage(String dataType, ArrayList timestamps, ArrayList values) throws IOException {
        FileOutputStream fileOutputStream = null;

        try {
//            ファイルの準備
            String filePath = getExternalFilesDir(null).toString() + "/" +
                    new SimpleDateFormat("yyyyMMdd_HHmmss").format(mRecordingDateTime) +
                    "_" + dataType + "_" + Build.MODEL.replaceAll(" ", "") + ".csv";

            fileOutputStream = new FileOutputStream(filePath, true);
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream, StandardCharsets.UTF_8);
            BufferedWriter bw = new BufferedWriter(outputStreamWriter);

//            ヘッダーの書き込み
            bw.write("Timestamp," + dataType);
            bw.newLine();

//            データの書き込み
            for(int i = 0; i < values.size(); i++) {
                bw.write(String.valueOf(timestamps.get(i)) + "," + String.valueOf(values.get(i)));
                bw.newLine();
            }

            bw.flush();
        } catch(Exception e) {
            Toast.makeText(getApplicationContext(), e.toString(), Toast.LENGTH_LONG).show();
        } finally {
            fileOutputStream.close();
        }
    }
}