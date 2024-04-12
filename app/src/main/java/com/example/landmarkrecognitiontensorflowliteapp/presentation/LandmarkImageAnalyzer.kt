package com.example.landmarkrecognitiontensorflowliteapp.presentation

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.landmarkrecognitiontensorflowliteapp.domain.Classification
import com.example.landmarkrecognitiontensorflowliteapp.domain.LandmarkClassifierListener
import com.example.landmarkrecognitiontensorflowliteapp.presentation.utils.centerCrop


class LandmarkImageAnalyzer(
    private val landmarkClassifierListener: LandmarkClassifierListener,
    private val onResults: (List<Classification>) -> Unit
) : ImageAnalysis.Analyzer {

    private var frameSkippedCounter = 0

    //call for each frame of Camera
    override fun analyze(image: ImageProxy) {

        //Process every 60th frame
        if (frameSkippedCounter % 60 == 0) {
            val rotationDegrees = image.imageInfo.rotationDegrees

            // The input image requires the 312 * 312 pixel
            val bitmap = image
                .toBitmap()
                .centerCrop(312, 312)

            val results = landmarkClassifierListener.classify(bitmap,rotationDegrees)

            //send back the result
            onResults(results)
        }
        frameSkippedCounter++

        //to indicate we have processed the image and now it can release it
        image.close()
    }


}