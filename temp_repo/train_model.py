import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- PATHS ----------------
DATASET_PATH = "dataset/train"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_food_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# ---------------- DATA AUGMENTATION + SPLIT ----------------
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 🔥 auto split
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ---------------- SAVE LABEL MAP ----------------
label_map = train_data.class_indices
joblib.dump(label_map, LABEL_MAP_PATH)

print("Label Map:", label_map)

# ---------------- MODEL (TRANSFER LEARNING) ----------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(len(label_map), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# ---------------- TRAIN ----------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Training Completed!")
print(f"Model saved at: {MODEL_PATH}")