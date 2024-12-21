package com.example.plantdisease;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final String TAG = "PlantDiseaseDetection";

    private ImageView resultImageView;
    private Interpreter tflite;
    private TextView resultTextView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultImageView = findViewById(R.id.resultImageView);
        Button selectImageBtn = findViewById(R.id.selectImageBtn);
        resultTextView = findViewById(R.id.resultTextView);


        // Load the TensorFlow Lite model
        try {
            tflite = new Interpreter(loadModelFile("cnn_model.tflite"));
        } catch (IOException e) {
            Log.e(TAG, "Error loading model", e);
        }

        // Handle button click for selecting an image
        selectImageBtn.setOnClickListener(v -> {
            Intent chooseImageIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(chooseImageIntent, PICK_IMAGE_REQUEST);
        });
    }

    // Load the TensorFlow Lite model as a MappedByteBuffer
    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        FileInputStream inputStream = new FileInputStream(getAssets().openFd(modelFileName).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = getAssets().openFd(modelFileName).getStartOffset();
        long declaredLength = getAssets().openFd(modelFileName).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            try {
                Uri imageUri = data.getData();
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                resultImageView.setImageBitmap(bitmap);
                classifyImage(bitmap);
            } catch (IOException e) {
                Log.e(TAG, "Error processing image", e);
            }
        }
    }

    // Classify the selected image
    private void classifyImage(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        float[][] result = runInference(resizedBitmap);

        // Find the class with the highest probability
        int maxIdx = 0;
        float maxProb = result[0][0];
        for (int i = 1; i < result[0].length; i++) {
            if (result[0][i] > maxProb) {
                maxProb = result[0][i];
                maxIdx = i;
            }
        }

        // Update TextView with the result
        String[] classNames = {"Early Blight", "Late Blight", "Healthy"};
        String resultText = "Predicted: " + classNames[maxIdx] + "\nProbability: " + maxProb;
        resultTextView.setText(resultText);

        Log.d("Classification Result", resultText);
    }


    // Run inference with TensorFlow Lite model
    private float[][] runInference(Bitmap bitmap) {
        // Preprocess the image into a 4D tensor (1, 224, 224, 3)
        float[][][][] input = preprocessImage(bitmap);
        float[][] output = new float[1][4]; // Adjust size based on your model's output
        tflite.run(input, output);
        return output;
    }

    // Preprocess the image into a 4D array
    private float[][][][] preprocessImage(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[][][][] result = new float[1][224][224][3];

        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int pixel = bitmap.getPixel(x, y);
                result[0][y][x][0] = (pixel >> 16 & 0xFF) / 255.0f; // Red
                result[0][y][x][1] = (pixel >> 8 & 0xFF) / 255.0f;  // Green
                result[0][y][x][2] = (pixel & 0xFF) / 255.0f;       // Blue
            }
        }

        return result;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}
