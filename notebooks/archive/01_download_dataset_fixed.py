from datasets import load_dataset
import os
from tqdm import tqdm

# Class mapping (fixed order)
CLASS_NAMES = [
    "Quartzity",
    "Live_Knot",
    "Marrow",
    "Resin",
    "Dead_Knot",
    "Knot_with_Crack",
    "Knot_missing",
    "Crack",
    "Blue_Stain",
    "Overgrown",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

dataset = load_dataset("iluvvatar/wood_surface_defects", split="train")

BASE_DIR = "C:/YOLOv8_OCHD_Project/datasets/wood_raw"
IMG_DIR = os.path.join(BASE_DIR, "images", "train")
LBL_DIR = os.path.join(BASE_DIR, "labels", "train")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
    # Save image
    img_path = os.path.join(IMG_DIR, f"{idx}.jpg")
    sample["image"].save(img_path)

    # Save labels
    label_path = os.path.join(LBL_DIR, f"{idx}.txt")

    with open(label_path, "w") as f:
        for obj in sample["objects"]:
            cls_name = obj["label"]
            if cls_name not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[cls_name]
            cx, cy, w, h = obj["bb"]  # already normalized YOLO format

            f.write(f"{cls_id} {cx} {cy} {w} {h}\n")

print("Dataset extraction completed successfully.")
