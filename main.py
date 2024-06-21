import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
    img_array = img_array.reshape((1, 28, 28, 1))  # Reshape for CNN model
    return img, img_array

# Streamlit App
st.title('MNIST Digit Classifier')

uploaded_image = st.file_uploader("Upload a digit image (MNIST format)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Preprocess the uploaded image
        user_image, img_array = preprocess_image(uploaded_image)

        # Display the processed images
        st.image(user_image, caption='Uploaded Image (Preprocessed)', use_column_width=True)

        # Make a prediction using the pre-trained model
        result = model.predict(img_array)
        predicted_class = np.argmax(result)

        st.success(f'Predicted Digit: {predicted_class}')

    except Exception as e:
        st.error(f'Error: {e}')
