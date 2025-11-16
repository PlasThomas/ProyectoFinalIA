# train_attitude.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Config ----------
DATA_DIR = "dataset"
IMG_SIZE = 160
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 8
EPOCHS_FINE = 12
LABELS = ['NEGATIVA','NEUTRA','POSITIVA']  # asegúrate que coincida con tu orden en directorio
MODEL_DIR = "saved_model"
# ---------------------------

# 1) Cargar datasets desde carpetas
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "validation"),
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Obtener el mapping por si acaso
class_names = train_ds.class_names
print("Clases detectadas (orden):", class_names)

# 2) Prefetch + performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 3) Calcular class weights para desbalanceo (opcional pero recomendado)
# Construimos lista de labels desde dataset
y_train = np.concatenate([y.numpy() for x,y in train_ds], axis=0)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: cw[i] for i in range(len(cw))}
print("class_weights:", class_weights)

# 4) Data augmentation (solo durante entrenamiento)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),  # +- ~3.5°
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.03, 0.03),
])

# 5) Preprocessing de MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def prep_layer(x, training=False):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)        # normaliza acorde al backbone
    if training:
        x = data_augmentation(x)
    return x

# Aplicar transformación en pipeline (map)
def map_fn(images, labels):
    images = tf.map_fn(lambda img: prep_layer(img, training=True), images, dtype=tf.float32)
    return images, labels

# Para entrenamiento aplicamos augmentación, para evaluación no
train_ds_aug = train_ds.map(lambda x,y: (prep_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)
val_ds = test_ds.map(lambda x,y: (prep_layer(x, training=False), y), num_parallel_calls=AUTOTUNE)

# 6) Construir modelo con Transfer Learning (MobileNetV2)
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 7) Callbacks
ckpt = callbacks.ModelCheckpoint("best_head.h5", monitor='val_accuracy', save_best_only=True, mode='max')
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 8) Entrenar solo la cabeza primero
history_head = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[ckpt, es, reduce_lr]
)

# 9) Fine-tuning: descongelar últimas capas del backbone
base.trainable = True
# Opcional: descongelar solo desde cierta capa
fine_tune_at = 100  # ajusta según tamaño del backbone y memoria
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

ckpt2 = callbacks.ModelCheckpoint("best_finetune.h5", monitor='val_accuracy', save_best_only=True, mode='max')
es2 = callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

history_fine = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=[ckpt2, es2, reduce_lr]
)

# 10) Evaluación final: predict sobre todo el test set
# Reconstruimos test X,y arrays (puedes hacerlo por batches)
y_true = []
y_pred = []
y_probs = []

for imgs, labels in val_ds:
    preds = model.predict(imgs)
    idxs = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(idxs.tolist())
    y_probs.extend(preds.tolist())

print("Classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

# 11) Guardar modelo completo
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "attitude_model.keras"))

print("Modelo guardado en:", os.path.join(MODEL_DIR, "attitude_model.keras"))
