package com.example.getpulse;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.wearable.activity.WearableActivity;
import android.support.wearable.view.BoxInsetLayout;
import android.support.wearable.view.WatchViewStub;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.common.api.ResultCallback;
import com.google.android.gms.wearable.MessageApi;
import com.google.android.gms.wearable.NodeApi;
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
    private final String TAG = MainActivity.class.getName();
    private final float GAIN = 0.9f;
    private TextView mTextView, tTextView;
    private Button startbutton, endbutton, autobutton;
    private SensorManager mSensorManager;
    private GoogleApiClient mGoogleApiClient;
    private String mNode;
    private float x=0,y=0,z=0, hr=0;
    int count = 0;
    int ppg = 0;
    ArrayList<Integer> values = new ArrayList<>();
    ArrayList<Float> timestamps = new ArrayList<>();
    Boolean is_recording = false, is_auto = false;
    private Date date;
    private long startTime;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mTextView = (TextView) findViewById(R.id.pulse);
        mTextView.setTextSize(20.0f);
        tTextView = (TextView) findViewById(R.id.time);
        tTextView.setTextSize(20.0f);
        startbutton = findViewById(R.id.start_button);
        endbutton = findViewById(R.id.end_button);
        autobutton = findViewById(R.id.auto_button);

        mSensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
        List<Sensor> sensorList = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        for (Sensor currentSensor : sensorList) {
            Log.d("List sensors", "Name: "+currentSensor.getName() + " /Type_String: " +currentSensor.getStringType()+ " /Type_number: "+currentSensor.getType());
        }
        //List sensors: Name: Heart Rate PPG Raw Data /Type_String: com.google.wear.sensor.ppg /Type_number: 65572

        mGoogleApiClient = new GoogleApiClient.Builder(this)
                .addApi(Wearable.API)
                .addConnectionCallbacks(new GoogleApiClient.ConnectionCallbacks() {
                    @Override
                    public void onConnected(Bundle bundle) {
                        Log.d(TAG, "onConnected");
//                        NodeApi.GetConnectedNodesResult nodes = Wearable.NodeApi.getConnectedNodes(mGoogleApiClient).await();
                        Wearable.NodeApi.getConnectedNodes(mGoogleApiClient).setResultCallback(new ResultCallback<NodeApi.GetConnectedNodesResult>() {
                            @Override
                            public void onResult(NodeApi.GetConnectedNodesResult nodes) {
                                //Nodeは１個に限定
                                if (nodes.getNodes().size() > 0) {
                                    mNode = nodes.getNodes().get(0).getId();
                                }
                            }
                        });
                    }
                    @Override
                    public void onConnectionSuspended(int i) {
                        Log.d(TAG, "onConnectionSuspended");
                    }
                })
                .addOnConnectionFailedListener(new GoogleApiClient.OnConnectionFailedListener() {
                    @Override
                    public void onConnectionFailed(ConnectionResult connectionResult) {
                        Log.d(TAG, "onConnectionFailed : " + connectionResult.toString());
                    }
                })
                .build();

        startbutton.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view){
                is_recording = true;
                timestamps.clear(); //arraylistの要素を削除
                values.clear();
                //現在日時の取得
                date = new Date();
                startTime = System.currentTimeMillis();
                //sensorValues = new StringBuilder();
                Log.d("CSV", "Start Sensing");
            }
        });

        endbutton.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view){
                is_recording = false;
                is_auto = false;
                Log.d("CSV", "Finish Sensing");

                try {
                    SaveToExternalStorage(timestamps, values);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        autobutton.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view){
                is_recording = true;
                is_auto = true;
                timestamps.clear(); //arraylistの要素を削除
                values.clear();
                //現在日時の取得
                date = new Date();
                startTime = System.currentTimeMillis();
                //sensorValues = new StringBuilder();
                Log.d("CSV", "Start Sensing");
            }
        });

    }

    private void SaveToExternalStorage(ArrayList timestamps, ArrayList values) throws IOException {
        FileOutputStream fileOutputStream = null;
        try {
            //フォーマット
            String day = new SimpleDateFormat("yyyyMMdd_HHmmss").format(date);

            String filePath = getExternalFilesDir(null).toString() + "/" + day + "_ticwatch.csv";
            Log.d("CSV", "filePath: " + filePath);
            fileOutputStream = new FileOutputStream(filePath, true);
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream, StandardCharsets.UTF_8);
            BufferedWriter bw = new BufferedWriter(outputStreamWriter);
            bw.write("time,pulse");
            bw.newLine();
            for(int i = 0; i < values.size(); i++) {
                bw.write(String.valueOf(timestamps.get(i)) + "," + String.valueOf(values.get(i)));
                bw.newLine();
            }
            bw.flush();
            Log.d("CSV", "Finish to write..");
        } catch(Exception ex) {
            Log.d("CSV", "Exception: " + ex.toString());
        }finally {
            fileOutputStream.close();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        //Sensor accSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        //mSensorManager.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_NORMAL);
        //Sensor hrSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE);
        //mSensorManager.registerListener(this, hrSensor, SensorManager.SENSOR_DELAY_NORMAL);
        Sensor ppgSensor = mSensorManager.getDefaultSensor(65572);
        mSensorManager.registerListener(this, ppgSensor, SensorManager.SENSOR_DELAY_FASTEST);
        mGoogleApiClient.connect();
    }
    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
        mGoogleApiClient.disconnect();
    }
    @Override
    public void onSensorChanged(SensorEvent event) {
        if(count>= 2) {
            count = 0;
            if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                //x = (x * GAIN + event.values[0] * (1 - GAIN));
                //y = (y * GAIN + event.values[1] * (1 - GAIN));
                //z = (z * GAIN + event.values[2] * (1 - GAIN));
                x = event.values[0];
                y = event.values[1];
                z = event.values[2];
                if (mTextView != null)
                    mTextView.setText(String.format("X : %f\nY : %f\nZ : %f" , x, y, z));
                //転送セット
                String SEND_DATA = x + "," + y + "," + z;
                if (mNode != null) {
                    Wearable.MessageApi.sendMessage(mGoogleApiClient, mNode, SEND_DATA, null).setResultCallback(new ResultCallback<MessageApi.SendMessageResult>() {
                        @Override
                        public void onResult(MessageApi.SendMessageResult result) {
                            if (!result.getStatus().isSuccess()) {
                                Log.d(TAG, "ERROR : failed to send Message" + result.getStatus());
                            }
                        }
                    });
                }
            }
            else if (event.sensor.getType() == Sensor.TYPE_HEART_RATE) {
                hr = event.values[0];
                if (mTextView != null)
                    mTextView.setText(String.format("心拍数 : %f" , hr));
            }
            else if (event.sensor.getType() == 65572) {
                ppg = (int) event.values[0];
                //float ppg = event.values[0];
                if(is_recording == true) {
                    //Log.d("CSV", Integer.toString(ppg));
                    long endTime = System.currentTimeMillis();
                    // カウント時間 = 経過時間 - 開始時間
                    long diffTime = (endTime - startTime);
                    //Log.d("CSV", Long.toString(diffTime));
                    SimpleDateFormat dataFormat = new SimpleDateFormat("mm:ss.SS", Locale.US);
                    tTextView.setText(dataFormat.format(diffTime));

                    timestamps.add((float)diffTime / 1000); //ArrayListに脈波データを追加
                    values.add(ppg);

                    if(is_auto == true) {
                        if(diffTime >= 30000) //何ミリ秒で計測終了するか
                            endbutton.performClick(); //endbuttonを押す
                    }
                }
                if (mTextView != null)
                    mTextView.setText(String.format("PPG : %d" , ppg));
            }
        }else count++;
    }
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }

}