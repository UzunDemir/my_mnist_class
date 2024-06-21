import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Загрузка предобученной модели
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Функция для предобработки изображения по пути к файлу
def preprocess_image_from_path(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Преобразование в grayscale
    img = img.resize((28, 28))  # Изменение размера до 28x28
    img_array = np.array(img) / 255.0  # Преобразование в массив и нормализация
    img_array = img_array.reshape((1, 28, 28, 1))  # Изменение формы для модели CNN
    return img, img_array

# Заголовок для Streamlit приложения
st.title('Классификатор рукописных цифр MNIST')

# Загрузка изображения с помощью виджета file_uploader
uploaded_file = st.file_uploader("Загрузите изображение цифры (формат MNIST)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Предобработка загруженного изображения
        user_image, img_array = preprocess_image_from_path(uploaded_file)

        # Отображение предобработанных изображений с помощью Streamlit
        st.subheader("Исходное изображение")
        st.image(user_image, caption='Исходное изображение', use_column_width=True, clamp=True)

        # Измененное изображение (28 * 28)
        resized_image = user_image.resize((28, 28))
        st.subheader("Измененное изображение (28x28)")
        st.image(resized_image, caption='Измененное изображение (28x28)', use_column_width=True, clamp=True)

        # Grayscale изображение
        grayscaled_image = resized_image.convert("L")
        st.subheader("Grayscale изображение")
        st.image(grayscaled_image, caption='Grayscale изображение', use_column_width=True, clamp=True)

        # Инвертированное изображение (текст белый, фон черный)
        inverted_image = 255 - np.array(grayscaled_image)
        st.subheader("Инвертированное изображение")
        st.image(inverted_image, caption='Инвертированное изображение', use_column_width=True, clamp=True)

        # Нормализованное изображение (делим на 255, чтобы значения были от 0 до 1)
        normalized_image = inverted_image / 255.0
        st.subheader("Нормализованное изображение")
        st.image(normalized_image, caption='Нормализованное изображение', use_column_width=True, clamp=True)

        # Измененная форма изображения
        reshaped_image = normalized_image.reshape((28, 28))
        st.subheader("Измененная форма изображения")
        st.image(reshaped_image, caption='Измененная форма изображения', use_column_width=True, clamp=True)

        # Предсказание с использованием предобученной модели
        result = model.predict(img_array)
        predicted_class = np.argmax(result)

        st.success(f'Предсказанная цифра: {predicted_class}')

    except Exception as e:
        st.error(f'Ошибка: {e}')
