# app.py
import streamlit as st
import numpy as np
import tensorflow as tf

# Загрузка обученной модели
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Функция для предсказания с помощью модели
def predict(image):
    # Предобработка изображения (нормализация и изменение размера)
    img = np.reshape(image.astype(np.float32) / 255.0, (1, 28, 28, 1))
    # Предсказание класса
    prediction = model.predict(img)
    # Возвращаем предсказанный класс (цифру от 0 до 9)
    return np.argmax(prediction)

# Основной код Streamlit
def main():
    st.title('Распознавание рукописных цифр (MNIST)')
    
    # Заголовок и описание
    st.markdown('Загрузите изображение рукописной цифры (черно-белое, 28x28 пикселей)')
    
    # Загрузка изображения
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Отображение изображения
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        # Предсказание по загруженному изображению
        prediction = predict(image)
        
        st.success(f'Предсказанная цифра: {prediction}')

if __name__ == '__main__':
    main()
