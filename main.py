import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained CNN model for MNIST digits
model = load_model('mnist_cnn_model.h5')

def preprocess_image(image):
    img = np.array(image.convert('L').resize((28, 28)))  # Convert, resize, and convert to grayscale
    img = img.reshape((1, 28, 28, 1)) / 255.0  # Reshape and normalize
    return img

def predict_digit(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return np.argmax(prediction)

# Streamlit App
st.title('MNIST Digit Recognizer')

uploaded_image = st.file_uploader("Upload an MNIST digit image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        digit = predict_digit(image)
        st.success(f'Prediction: {digit}')
