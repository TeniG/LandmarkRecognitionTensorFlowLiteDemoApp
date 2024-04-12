package com.example.landmarkrecognitiontensorflowliteapp.domain

import android.content.Context
import android.graphics.Bitmap
import android.view.Surface
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.lang.IllegalStateException

/*
threshold: Probability of match, Staring from which we will consider for result
maxResults: The number of matches to be shown as result.
 */
class TfLiteLandmarkClassifier(
    private val context: Context,
    private val threshold: Float = 0.5f,
    private val maxResults: Int = 1
) : LandmarkClassifierListener {

    private var imageClassifier: ImageClassifier? = null

    private fun setUpClassifier() {
        val baseOptions = BaseOptions.builder()
            .setNumThreads(2)
            .build()

        val imageClassifierOptions = ImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(maxResults)
            .setScoreThreshold(threshold)
            .build()


        try {

            imageClassifier = ImageClassifier.createFromFileAndOptions(
                context,
                "landmark.tflite",
                imageClassifierOptions
            )

        } catch (e: IllegalStateException) {
            e.printStackTrace()
        }
    }

    override fun classify(bitmap: Bitmap, rotation: Int): List<Classification> {
        if (imageClassifier == null) {
            setUpClassifier()
        }

        // Create preprocessor for the image.
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(rotation/90))
            .build()
        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build()

        // Preprocess the image and convert it into a TensorImage for classification.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        // Classify the input image
        val result = imageClassifier?.classify(tensorImage, imageProcessingOptions)

        return result?.flatMap { classification ->
            classification.categories.map { category ->
                Classification(name = category.displayName, score = category.score)
            }
        }?.distinctBy { it.name } ?: emptyList()
    }

    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        return when (rotation) {
            Surface.ROTATION_270 -> ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_90 -> ImageProcessingOptions.Orientation.TOP_LEFT
            Surface.ROTATION_180 -> ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            else -> ImageProcessingOptions.Orientation.RIGHT_TOP

        }
    }

}