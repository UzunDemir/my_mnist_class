import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка предобученной модели
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Функция для предобработки загруженного изображения
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Преобразование в grayscale
    img = img.resize((28, 28))  # Изменение размера до 28x28
    img_array = np.array(img) / 255.0  # Преобразование в массив и нормализация
    img_array = img_array.reshape((1, 28, 28, 1))  # Изменение формы для модели CNN
    return img, img_array

# Streamlit приложение
st.title('Классификатор рукописных цифр MNIST')

uploaded_image = st.file_uploader("Загрузите изображение цифры (формат MNIST)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Предобработка загруженного изображения
        user_image, img_array = preprocess_image(uploaded_image)
# if uploaded_image is not None:
#             # Preprocessed the data like the example
#             user_image = Image.open(uploaded_image)
#             fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

        # Отображение обработанных изображений с помощью Matplotlib
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

        # Скрытие меток осей для всех подграфиков
        for axis in ax.flat:
            axis.set_xticks([])
            axis.set_yticks([])

        # Исходное изображение
        ax[0, 0].set_title("Original Image")
        ax[0, 0].imshow(user_image)
        # ax[0, 0].set_title("Исходное изображение")
        # ax[0, 0].imshow(user_image, cmap='gray')  # Используем cmap='gray' для grayscale изображений

        # Измененное изображение (28 * 28)
        resized_image = user_image.resize((28, 28))
        ax[0, 1].set_title("Измененное изображение")
        ax[0, 1].imshow(resized_image, cmap='gray')

        # Grayscale изображение
        grayscaled_image = resized_image.convert("L")
        ax[0, 2].set_title("Grayscale изображение")
        ax[0, 2].imshow(grayscaled_image, cmap="gray")

        # Инвертированное изображение (текст белый, фон черный)
        inverted_image = 255 - np.array(grayscaled_image)
        ax[1, 0].set_title("Инвертированное изображение")
        ax[1, 0].imshow(inverted_image, cmap="gray")

        # Нормализованное изображение (делим на 255, чтобы значения были от 0 до 1)
        normalized_image = inverted_image / 255.0
        ax[1, 1].set_title("Нормализованное изображение")
        ax[1, 1].imshow(normalized_image, cmap="gray")

        # Измененное форма изображение
        reshaped_image = normalized_image.reshape((28, 28))
        ax[1, 2].set_title("Измененная форма изображения")
        ax[1, 2].imshow(reshaped_image, cmap="gray")

        st.pyplot(fig)

        # Предсказание с использованием предобученной модели
        # result = model.predict(img_array)
        # predicted_class = np.argmax(result)

        # st.success(f'Предсказанная цифра: {predicted_class}')

        predicted_label = model.predict(reshaped_image)
        predicted_label = predicted_label.argmax(axis=1)

            # st.subheader(f"True Value: {true_label}")
        st.subheader(f"Predicted Value: {predicted_label.tolist()[0]}")

    except Exception as e:
        st.error(f'Ошибка: {e}')
