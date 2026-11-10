import os
import shutil
import random

# ==============================
# SETTINGS
# ==============================
DATASET_DIR = "dataset"          # original folders (pizza, burger...)
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

SPLIT_RATIO = 0.8   # 80% train, 20% validation

# ==============================
# CREATE TRAIN & VAL FOLDERS
# ==============================
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ==============================
# LOOP THROUGH EACH CLASS
# ==============================
for class_name in os.listdir(DATASET_DIR):

    class_path = os.path.join(DATASET_DIR, class_name)

    # skip train & val folders if already exist
    if not os.path.isdir(class_path) or class_name in ["train", "val"]:
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    # create class folders inside train & val
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

    # move images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, class_name, img)
        shutil.move(src, dst)

    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(VAL_DIR, class_name, img)
        shutil.move(src, dst)

    print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")

print("\n🎉 Dataset successfully split into TRAIN and VAL folders!")