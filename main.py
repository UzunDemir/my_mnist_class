import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Загрузка модели
model = load_model('mnist_cnn_mod.h5')

# Функция для предобработки изображения
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((28, 28))
    img = img.convert('L')  # Преобразование в оттенки серого
    img_array = np.array(img) / 255.0  # Нормализация
    img_array = img_array.reshape((1, 28, 28, 1))  # Добавление размерности канала
    return img_array

# Функция для загрузки и предсказания
def predict_digit(image_path):
    img_array = preprocess_image(image_path)
    
    # Предсказание класса с использованием модели
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    return predicted_class

# Пример использования
image_path = 'images.png'  # Укажите путь к вашему изображению с цифрой
predicted_digit = predict_digit(image_path)
print(f'Predicted Digit: {predicted_digit}')
