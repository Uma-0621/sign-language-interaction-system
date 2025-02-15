import cv2
import numpy as np
import tensorflow as tf
import time
from cvzone.HandTrackingModule import HandDetector

# Load the saved model (from local path)
model = tf.keras.models.load_model('sign_language_model77.h5')

# Create the hand detector
detector = HandDetector(maxHands=1)

# Set up the webcam
cap = cv2.VideoCapture(0)
imgSize = 64  # Size used for training

# Load the label encoder for class mapping (ensure this is the same one used during training)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Example labels used for training (you should replace this with the actual labels from training)
label_encoder.fit(["Hello", "Gud Job", "I love you", "No", "Okay", "Yes", "You"])

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))
        # Prepare image for prediction (normalize and expand dimensions)
        imgArray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        imgArray = np.array(imgArray) / 255.0  # Normalize pixel values
        imgArray = np.expand_dims(imgArray, axis=0)  # Add batch dimension
        # Test on an image from the dataset    
        # Make prediction
        predictions = model.predict(imgArray)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Only display prediction if confidence is above threshold (e.g., 70%)
        if confidence > 0.7:
            class_label = label_encoder.inverse_transform([predicted_class])[0]
            cv2.putText(img, f"Prediction: {class_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Prediction: Not Confident", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Optional: Display confidence
        cv2.putText(img, f"Confidence: {confidence*100:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("q"):  # Press 'q' to exit
        break
cap.release()
cv2.destroyAllWindows()
