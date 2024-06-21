import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('mnist_cnn_model.h5')

def preprocess_image(img):
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применение пороговой обработки
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Инверсия цветов (если нужно)
    thresh = 255 - thresh
    
    # Изменение размера до 28x28
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Нормализация значений пикселей
    normalized = resized.astype('float32') / 255.0
    
    # Решейп до формы (1, 28, 28, 1) для подачи в модель
    img_array = normalized.reshape((1, 28, 28, 1))
    
    return img_array

def rec_digit(img_path):
    # Загрузка изображения
    img = cv2.imread(img_path)
    
    # Предобработка изображения
    img_array = preprocess_image(img)
    
    # Предсказание с использованием модели
    result = model.predict(img_array)
    
    # Получение предсказанного класса
    predicted_class = np.argmax(result)
    
    return str(predicted_class)

# Пример использования
img_path = 'path/to/your/image.jpg'
prediction = rec_digit(img_path)
print(f"Predicted digit: {prediction}")
