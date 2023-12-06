import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the pre-trained model
model = load_model("best_model_res_classes_2.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face area
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))

        # Convert grayscale to RGB (3 channels)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        # Preprocess the face for the model
        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face /= 255

        # Predict the emotion
        prediction = model.predict(face)
        max_index = np.argmax(prediction[0])
        emotions = ['happy', 'sad']  # Add more emotions as per your model
        predicted_emotion = emotions[max_index]

        # Put text of emotion on the rectangle
        cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Facial Emotion Recognition', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
