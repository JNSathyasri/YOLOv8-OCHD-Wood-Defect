import cv2
import os
import random

BASE_DIR = "C:/YOLOv8_OCHD_Project/datasets/wood_raw"

IMG_DIR = os.path.join(BASE_DIR, "images", "train")
LBL_DIR = os.path.join(BASE_DIR, "labels", "train")

# Pick 5 random images
images = random.sample(os.listdir(IMG_DIR), 5)

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())

                # YOLO normalized -> pixel coordinates
                cx, cy = int(cx * w), int(cy * h)
                bw, bh = int(bw * w), int(bh * h)

                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    str(int(cls)),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("Label Check", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
