import numpy as np
from PIL import Image
import io
import tensorflow as tf


IMG_SIZE = 160
LABELS = ['NEGATIVA','NEUTRA','POSITIVA']


# carga del modelo
def load_model(path):
    # usa tf.keras.models.load_model para cargar .keras o .h5
    model = tf.keras.models.load_model(path)
    return model


# convierte bytes a PIL
def pil_from_bytes(image_bytes: bytes):
    return Image.open(io.BytesIO(image_bytes))


# preprocesado identico al usado en entrenamiento con MobileNetV2
def preprocess_pil_image(pil_img: Image.Image):
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr