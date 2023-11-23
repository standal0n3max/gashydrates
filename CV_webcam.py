import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image_class(model, frame):
    # Preprocess the frame
    img = cv2.resize(frame, (150, 150))  # Resize frame to match model's input size
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

# Load the trained model
model = tf.keras.models.load_model('hydrate_model.h5')

# Access the webcam
video_capture = cv2.VideoCapture(0)  # '0' usually represents the default webcam

# Check if the webcam is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open webcam")
    exit()


# Read and display frames in real-time
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error reading frame from webcam")
        break

    # Perform prediction using the loaded model
    predicted_class = predict_image_class(model, frame)

    # Add predicted class label to the frame
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
