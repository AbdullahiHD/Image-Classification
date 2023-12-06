import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Load the pre-trained model
model = load_model("best_model_res_classes_4.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect emotion
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (224, 224))

        # Convert grayscale to RGB
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

        img_pixels = image.img_to_array(roi_color)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ['happy', 'sad', 'surprise', 'angry']  # Adjust as per your model
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame
# Function to update frames
def update_frame():
    global cap
    ret, frame = cap.read()
    if ret:
        frame = detect_emotion(frame)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)
    label.after(10, update_frame)

# Setup the main window
root = tk.Tk()
root.title("Live Emotion Detection")

# Label to display the video
label = tk.Label(root)
label.pack()

# Initialize video capture for webcam
cap = cv2.VideoCapture(0)

# Start the update function in a separate thread
thread = threading.Thread(target=update_frame)
thread.start()

# Run the application
root.mainloop()

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
