from datasets import load_dataset
import os
import shutil

# Load dataset
dataset = load_dataset("iluvvatar/wood_surface_defects")

base_dir = "C:/YOLOv8_OCHD_Project/datasets/wood_raw"
os.makedirs(base_dir, exist_ok=True)

def save_split(split_name):
    split = dataset[split_name]
    img_dir = os.path.join(base_dir, "images", split_name)
    lbl_dir = os.path.join(base_dir, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i, sample in enumerate(split):
        # Save image
        img_path = os.path.join(img_dir, f"{i}.jpg")
        sample["image"].save(img_path)

        # Save YOLO label
        label_path = os.path.join(lbl_dir, f"{i}.txt")
        with open(label_path, "w") as f:
            for box, cls in zip(sample["bboxes"], sample["labels"]):
                x, y, w, h = box
                f.write(f"{cls} {x} {y} {w} {h}\n")

for split in dataset.keys():
    save_split(split)

print("Dataset downloaded successfully.")