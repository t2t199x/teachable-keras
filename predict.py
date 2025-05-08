from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load model
model = load_model("keras_model.h5", compile=False)

# Load class labels
class_names = open("labels.txt", "r").readlines()

# Load image
image = Image.open("test.jpg")  # ảnh cần dự đoán
image = image.convert("RGB")
image = image.resize((224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)

# Chuẩn hóa và reshape
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.expand_dims(normalized_image_array, axis=0)

# Dự đoán
prediction = model.predict(data)
index = np.argmax(prediction)
print("Kết quả:", class_names[index].strip())
