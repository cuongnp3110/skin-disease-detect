from flask import Flask, render_template, request
from flask_restful import Api, Resource

import keras.utils as image
import os
import numpy as np
import pandas as pd
from keras.models import load_model
import shutil
import cv2 as cv2

# Load mô hình từ file
# Hàm để thay đổi kích thước ảnh
def resize_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (100, 75))
    return resized_image


# Đường dẫn tới ảnh cần dự đoán
image_path = r'C:\Users\ADMIN\Desktop\TrungPrj\App\ISIC_0027008.jpg'

try:
    with open(image_path, 'rb') as f:
        pass
except FileNotFoundError:
    print("File not found:", image_path)
except Exception as e:
    print("Error reading file:", e)

# Đọc ảnh bằng OpenCV và kiểm tra xem ảnh có bị rỗng sau khi đọc không
image = cv2.imread(image_path)
if image is None:
    print("Error loading image from file:", image_path)
else:
    print("Image loaded successfully.")

# Load mô hình từ file
model = load_model('model-ham10000.h5')

# Thay đổi kích thước của ảnh
resized_image = resize_image(image_path)

# Mở rộng chiều cho batch (batch_size = 1)
resized_image = np.expand_dims(resized_image, axis=0)

# Sử dụng mô hình để dự đoán
predictions = model.predict(resized_image)

# Chuyển đổi dự đoán thành tên của loại bệnh tương ứng
lesion_type_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions ',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# In kết quả dự đoán
predicted_class = np.argmax(predictions)
predicted_lesion_type = lesion_type_dict[predicted_class]
print("Predicted Lesion Type:", predicted_lesion_type)