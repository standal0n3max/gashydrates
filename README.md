# CNN.ipynb
# Hydrates Detection Model

This repository contains code for a machine learning model that detects the presence of "hydrates" in images using convolutional neural networks (CNNs) built with Keras and TensorFlow.

## Requirements

- Python 3.x
- Libraries:
  - Keras
  - TensorFlow
  - Split-folders
  - Matplotlib
  - NumPy

## Installation

Install the required libraries using pip:

```pip install keras tensorflow split-folders matplotlib numpy==1.22.3```

Dataset Splitting

The dataset has been split into training, validation, and test sets using the split-folders library. The dataset is organized as follows:

Training Set: Hydrates_split/train
Validation Set: Hydrates_split/val
Test Set: Hydrates_split/test

## Model Architecture

The model architecture utilizes a pre-trained ResNet50V2 convolutional base followed by additional fully connected layers for classification. The model summary is provided in the code.

## Training

The model has been trained using the training and validation sets for 30 epochs. Checkpoints have been implemented to save the best model based on validation loss.

## Evaluation

The trained model achieved an accuracy of 97.92% on the test data.

## Usage

### Predicting on Images
You can predict the presence of "hydrates" in images using the provided predict_image_class function. Example usage:
```
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
hydrdrate_model = load_model('hydrate_model.h5')

def predict_image_class(model, image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1

    # Make the prediction
    prediction = model.predict(img_array)

    # Decode the prediction
    if prediction[0][0] > 0.5:
        return "hydrates_yes"
    else:
        return "hydrates_no"

image_path = "path/to/your/image.jpg"
predicted_class = predict_image_class(hydrdrate_model, image_path)
print(f"The predicted class is: {predicted_class}")
```

### Real-time Detection
To perform real-time detection and trigger an alarm when "hydrates_yes" is detected in a video stream, integrate the provided code into your video stream handling process. Ensure to replace the get_next_frame() function with the logic to receive frames in real-time from your video source.

```
# Placeholder function to simulate getting the next frame in real-time
def get_next_frame():
    # Replace this with the code to fetch the next frame from your video stream
    # This function should return a frame (image) for processing
    pass

while True:
    # Get the next frame from the real-time stream
    frame = get_next_frame()

    # Process the frame using your trained model
    predicted_class = predict_image_class(hydrdrate_model, frame)  # Use the function defined earlier

    # Check for the 'hydrates_yes' class and trigger an alarm
    if predicted_class == 'hydrates_yes':
        print("ALARM! 'hydrates_yes' detected.")
        # Add your alarm code here (e.g., sound an alarm, display a message, etc.)

    # Perform other processing or visualization with the predicted class as needed
```

# CV_webcam
## Real-time Image Classification using TensorFlow and OpenCV

This repository contains a Python script that utilizes TensorFlow and OpenCV to perform real-time image classification using a pre-trained model.

## Overview

The script uses a pre-trained neural network model to predict whether an image captured from the webcam belongs to a particular class ("hydrates_yes" or "hydrates_no"). The model was trained to identify hydration-related images.

## Requirements

- Conda (Anaconda or Miniconda)
- Python 3.x
- Environment Setup

## Environment Setup

1. Clone the repository:
```git clone https://github.com/standal0n3max/gashydrates.git```

2. Navigate to the project directory:
```cd gashydrates```

3. Create the Conda environment from environment.yml:
```conda env create -n hydra -f environment.yml```
This command creates a Conda environment named image_classification_env with the required dependencies listed in environment.yml.

4. Activate the environment:
```conda activate hydra```

## Usage

Run the script:
```python CV_webcam.py```

The script accesses the default webcam (video_capture = cv2.VideoCapture(0)) and displays the real-time feed with predictions overlaying each frame.
Exit the program:
Press 'q' to exit the webcam feed window.

## Code Explanation

- The script initializes the webcam and loads a pre-trained TensorFlow model (hydrate_model.h5).
- It continuously captures frames from the webcam and performs predictions using the loaded model.
- Predicted class labels are overlaid on the frames displayed in real-time using OpenCV.
- Press 'q' to exit the webcam feed window and stop the script.
