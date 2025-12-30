import os
import random
import shutil

RAW_BASE = "C:/YOLOv8_OCHD_Project/datasets/wood_raw"
FINAL_BASE = "C:/YOLOv8_OCHD_Project/datasets/wood"

IMG_SRC = os.path.join(RAW_BASE, "images", "train")
LBL_SRC = os.path.join(RAW_BASE, "labels", "train")

IMG_TRAIN = os.path.join(FINAL_BASE, "images", "train")
IMG_VAL = os.path.join(FINAL_BASE, "images", "val")
LBL_TRAIN = os.path.join(FINAL_BASE, "labels", "train")
LBL_VAL = os.path.join(FINAL_BASE, "labels", "val")

os.makedirs(IMG_TRAIN, exist_ok=True)
os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_TRAIN, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

files = [f for f in os.listdir(IMG_SRC) if f.endswith(".jpg")]
random.shuffle(files)

split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

def move(files, img_dst, lbl_dst):
    for f in files:
        shutil.copy(
            os.path.join(IMG_SRC, f),
            os.path.join(img_dst, f)
        )
        shutil.copy(
            os.path.join(LBL_SRC, f.replace(".jpg", ".txt")),
            os.path.join(lbl_dst, f.replace(".jpg", ".txt"))
        )

move(train_files, IMG_TRAIN, LBL_TRAIN)
move(val_files, IMG_VAL, LBL_VAL)

print(f"Train images: {len(train_files)}")
print(f"Val images: {len(val_files)}")
