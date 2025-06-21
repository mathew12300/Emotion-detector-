
# Emotion Detection System

## **Project Description**

This project is an **Emotion Detection System** built using Python and Convolutional Neural Networks (CNN). It leverages deep learning to classify human facial expressions into seven emotion categories in real-time using a webcam feed.

The system uses grayscale face images to train a CNN model, then predicts emotions such as Angry, Happy, Sad, etc., providing a practical tool for emotion recognition useful in areas like user experience, healthcare, and human-computer interaction.

---
# (check.py is optional )
## **Features**

1. **Real-Time Emotion Detection:**

   * Detects and classifies emotions from live webcam video feed instantly.

2. **CNN-Based Deep Learning Model:**

   * Trains a convolutional neural network on facial expression datasets.

3. **Model Training and Evaluation:**

   * Includes training with accuracy and loss visualization to monitor model performance.

4. **Simple Deployment:**

   * Model weights are saved and can be loaded later for real-time emotion prediction without retraining.

5. **Emotion Categories Supported:**

   * Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.

---

## **Prerequisites**

### **Software Requirements**

1. **Python 3.7 or Higher**

   * Ensure Python is installed and added to your system PATH.

2. **Required Python Libraries:**

   * `tensorflow`
   * `numpy`
   * `opencv-python`
   * `matplotlib`

   Install dependencies via pip:

   ```
   pip install tensorflow numpy opencv-python matplotlib
   ```

3. **Haarcascade File for Face Detection**

   * `haarcascade_frontalface_default.xml` is required for face localization and should be placed in the project directory.

4. **Code Editor**

   * Any Python-supporting IDE such as PyCharm, VS Code, or Jupyter Notebook.

---

### **Hardware Requirements**

1. **Webcam:**

   * A functional webcam for real-time video capture.

2. **System Configuration:**

   * Recommended minimum: 4GB RAM, dual-core CPU for smooth training and detection.

---

## **Project Structure**

* **data/train/** - Training images organized in folders named after emotion classes.
* **data/test/** - Validation images organized similarly.
* **haarcascade\_frontalface\_default.xml** - Haarcascade XML for face detection.
* **model.h5** - Saved model weights after training.
* **main.py** - Main Python script to train or run detection.

---

## **Execution Steps**

### Step 1: Prepare Dataset

* Organize your training and validation images in `data/train` and `data/test` folders with subfolders for each emotion label.

### Step 2: Install Dependencies

* Use the command to install required Python libraries as mentioned above.

### Step 3: Train the Model

* Run the script in **training mode** to start training the CNN model on your dataset.
* The training process will output accuracy and loss graphs and save the trained model weights as `model.h5`.

### Step 4: Run Real-Time Emotion Detection

* Switch the script to **display mode** to use your webcam for live emotion detection.
* Detected faces will be bounded with rectangles and labeled with predicted emotions in real-time.

### Step 5: Exit Application

* Press **'q'** to close the webcam feed and terminate the program.

---

## **Emotion Classes**

| Index | Emotion   |
| ----- | --------- |
| 0     | Angry     |
| 1     | Disgusted |
| 2     | Fearful   |
| 3     | Happy     |
| 4     | Neutral   |
| 5     | Sad       |
| 6     | Surprised |

---

## **Notes**

* Ensure `haarcascade_frontalface_default.xml` is in the same directory as your script.
* The dataset should be grayscale images of size 48x48 pixels for optimal training.
* Model training may take time depending on hardware capabilities.
* Adjust batch size and epochs in the script as needed for your dataset size and compute power.
* Webcam index may need to be adjusted if multiple cameras are connected (default is 0).

---

## **Contributing**

Contributions, suggestions, and improvements are welcome! Feel free to open issues or submit pull requests.

---

## **License**

This project is licensed under the MIT License.

---

## **Acknowledgments**

* The Haarcascade face detector from OpenCV.
* TensorFlow and Keras libraries for deep learning framework.
* Public datasets and inspiration from facial expression recognition research.



