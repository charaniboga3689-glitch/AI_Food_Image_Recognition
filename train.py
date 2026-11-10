"""
Train a food image classifier using transfer learning with MobileNetV2.
Dataset: Food-101 (downloaded automatically by TensorFlow Datasets).
Run: python train.py
"""
import json
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 3
OUTPUT_DIR = Path("model")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading Food-101 dataset. First run downloads the dataset (several GB).")
(train_ds, validation_ds), info = tfds.load(
    "food101",
    split=["train", "validation"],
    as_supervised=True,
    with_info=True
)
class_names = info.features["label"].names
(OUTPUT_DIR / "class_names.json").write_text(
    json.dumps(class_names, indent=2), encoding="utf-8"
)
num_classes = len(class_names)

def prepare(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds
    .shuffle(3000)
    .map(prepare, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))
validation_ds = (validation_ds
    .map(prepare, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
], name="data_augmentation")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_DIR / "food_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )
]

print("\nStage 1: training the classification head...")
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

print("\nStage 2: fine-tuning the last MobileNetV2 layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

model.save(OUTPUT_DIR / "food_model.keras")
print(f"\nTraining complete. Model saved to {OUTPUT_DIR / 'food_model.keras'}")
print(f"Number of food categories: {num_classes}")
