# YOLOv8-OCHD: Wood Surface Defect Detection

This repository presents an implementation and ablation study of an improved YOLOv8-based object detection framework for wood surface defect detection, inspired by recent research on lightweight and multi-scale detection architectures.

---

## 📌 Project Overview

The objective of this project is to analyze the impact of progressive architectural enhancements on YOLOv8 for detecting surface defects in wood images. The study evaluates multiple model variants through controlled experiments and comparative analysis.

Following the design principles described in the paper, ODConv, C2f RVB, Haar wavelet-based downsampling, and DyHead are implemented using YOLOv8-compatible architectural enhancements. Due to framework and computational constraints, these components are realized through functionally equivalent modules rather than explicit custom operators.

---

## 🧠 Model Variants

The following models are implemented and evaluated:

- **Baseline**: YOLOv8n
- **O**: Output-scale enhanced model (ODConv)
- **OC**: Output-scale with contextual modeling (C2f RVB)
- **OCH**: Output-scale, contextual modeling, and hierarchical feature fusion (HWD)
- **OCHD**: OCH with DyHead-style multi-scale detection (DyHead)

Each variant is trained and evaluated independently, and their performance is compared using standard object detection metrics.

---

## 📂 Dataset

The dataset used in this project is obtained from Hugging Face:

**Large-Scale Wood Surface Defect Dataset**
- Approximately 20,000 images
- 10 defect categories
- Bounding-box annotations

Dataset source:  
https://huggingface.co/datasets/iluvvatar/wood_surface_defects

Due to dataset size constraints, the dataset is not included in this repository. Dataset download and preprocessing are handled programmatically within the notebooks.

---

## 🧪 Experiments and Notebooks

All experiments were conducted using executed Jupyter notebooks. The notebooks include full outputs such as training logs, validation results, and evaluation metrics.

| Notebook | Description |
|--------|-------------|
| `01_dataset_preparation.ipynb` | Dataset download and preprocessing |
| `02_baseline_yolov8n.ipynb` | Baseline YOLOv8n training |
| `03_O_OC_training.ipynb` | Training of O and OC variants |
| `04_OCH_training.ipynb` | Training of OCH variant |
| `05_OCHD_training.ipynb` | Training of OCHD variant |

---

## ⚙️ How to Run

### Environment Setup
```bash
pip install ultralytics torch torchvision datasets
