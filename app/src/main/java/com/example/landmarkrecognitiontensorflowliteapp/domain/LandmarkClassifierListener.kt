package com.example.landmarkrecognitiontensorflowliteapp.domain

import android.graphics.Bitmap

interface LandmarkClassifierListener {
    fun classify(bitmap: Bitmap, rotation: Int) : List<Classification>
}