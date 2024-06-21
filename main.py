import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

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
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

        # Hide the ticks
        for axis in ax.flat:
            axis.set_xticks([])
            axis.set_yticks([])

        # Original Image
        ax[0, 0].set_title("Original Image")
        ax[0, 0].imshow(user_image)

        # Resized Image (28 * 28)
        resized_image = user_image.resize((28, 28))
        ax[0, 1].set_title("Resized Image")
        ax[0, 1].imshow(resized_image)

        # Grayscale Image
        grayscaled_image = resized_image.convert("L")
        ax[0, 2].set_title("Grayscaled Image")
        ax[0, 2].imshow(grayscaled_image, cmap="gray")

        # Invert the image (so the text is white, and background is black)
        inverted_image = 255 - np.array(grayscaled_image)
        ax[1, 0].set_title("Inverted Image")
        ax[1, 0].imshow(inverted_image, cmap="gray")

        # Normalize the image (divided by 255 to make them between 0 - 1)
        normalized_image = inverted_image / 255.0
        ax[1, 1].set_title("Normalized Image")
        ax[1, 1].imshow(normalized_image, cmap="gray")

        # Reshaped image
        reshaped_image = normalized_image.reshape((28, 28))
        ax[1, 2].set_title("Reshaped Image")
        ax[1, 2].imshow(reshaped_image, cmap="gray")

        st.pyplot(fig)

        # # Make a prediction using the pre-trained model
        # result = model.predict(img_array)
        # predicted_class = np.argmax(result)

        # st.success(f'Predicted Digit: {predicted_class}')
        # The actual data we're going to use
        reshaped_image = normalized_image.reshape((1, 28 * 28))

        # true_label = st.selectbox(
        #     "What digit does your image represent?", [i for i in range(10)]
        # )
        predicted_label = model.predict(reshaped_image)
        predicted_label = predicted_label.argmax(axis=1)

        # st.subheader(f"True Value: {true_label}")
        st.subheader(f"Predicted Value: {predicted_label.tolist()[0]}")
        st.divider()
        
    except Exception as e:
        st.error(f'Error: {e}')
