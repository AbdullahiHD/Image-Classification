import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the previously trained model
model = load_model("best_model.h5")

# Load the Haar cascade file for face detection
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from webcam
    ret, test_img = cap.read()
    if not ret:
        continue

    # Convert image to grayscale for face detection
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Detect faces in image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for x, y, w, h in faces_detected:
        # Draw rectangle around detected faces
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

        # Crop the detected face region
        roi_gray = gray_img[y : y + w, x : x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))

        # Convert cropped face image to array and preprocess it for the model
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Predict the emotion of the face
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        # Define the list of emotions and select the predicted one
        emotions = ("happy", "sad")
        predicted_emotion = emotions[max_index]

        # Display the predicted emotion on the image
        cv2.putText(
            test_img,
            predicted_emotion,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Resize and display the image
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Facial emotion analysis ", resized_img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) == ord("q"):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
