import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import math
from PIL import Image

# Load the pre-trained CNN model for MNIST digits
model = load_model('mnist_cnn_model.h5')

def calculate_center_of_mass(img):
    # Function to calculate center of mass of a grayscale image
    rows, cols = img.shape
    total_mass = np.sum(img)
    if total_mass == 0:
        return cols // 2, rows // 2  # Return center if no mass found

    cy = np.sum(np.arange(rows).reshape(-1, 1) * img) / total_mass
    cx = np.sum(np.arange(cols).reshape(1, -1) * img) / total_mass

    return int(cx), int(cy)

def preprocess_image(image):
    # Function to preprocess the uploaded image
    img = np.array(image.convert('L'))  # Convert to grayscale numpy array
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    img = img / 255.0  # Normalize to [0, 1]
    img = img.reshape((1, 28, 28, 1))  # Reshape for model input
    return img

def rec_digit(img):
    # Function to recognize the digit from preprocessed image
    gray = 255 - img
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
    
    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = calculate_center_of_mass(gray)
    shifted = np.roll(gray, shiftx, axis=1)
    gray = np.roll(shifted, shifty, axis=0)

    gray = gray.reshape(-1, 28, 28, 1)  # Reshape for model input
    out = str(np.argmax(model.predict(gray)))  # Predict the digit
    return out

# Streamlit App
st.title('MNIST Digit Recognizer')

uploaded_image = st.file_uploader("Upload an MNIST digit image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read image
    image = Image.open(uploaded_image).convert('L')
    
    # Display image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        # Preprocess and predict
        prediction = rec_digit(np.array(image))
        st.success(f'Prediction: {prediction}')
