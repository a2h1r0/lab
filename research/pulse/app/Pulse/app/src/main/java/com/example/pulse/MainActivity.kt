package com.example.pulse

import android.R
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {
    private val handler = Handler(Looper.getMainLooper())
    private var r: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var count = 0
        r = Runnable {
            val imageView: ImageView = findViewById(R.id.imageBluePaint)
            imageView.setImageResource(R.drawable.bluepaint)
            count++
            Log.d("count", count.toString())
            r?.let { handler.postDelayed(it, 1000) }
        }
        handler.post(r!!)
    }
}