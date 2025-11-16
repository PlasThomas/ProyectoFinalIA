from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_image(model, img_path, IMG_SIZE=160, class_names=['NEGATIVA','NEUTRA','POSITIVA']):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx])

# uso
#model = tf.keras.models.load_model("saved_model/attitude_model.keras")
#model = tf.keras.models.load_model("alt_models/best_finetune.h5")
model = tf.keras.models.load_model("alt_models/best_head.h5")

label, conf = predict_image(model, "foto1.jpeg")
print(label, conf)
label, conf = predict_image(model, "foto2.jpeg")
print(label, conf)
label, conf = predict_image(model, "foto3.jpeg")
print(label, conf)
label, conf = predict_image(model, "foto4.jpeg")
print(label, conf)