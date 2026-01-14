# Fabella Detection Project

This project provides a complete pipeline for detecting the **fabella** bone in medical X-ray images using YOLOv11/v12 with **Oriented Bounding Boxes (OBB)**. The workflow covers DICOM preprocessing, data cleaning, manual labeling, dataset preparation, and model training/testing.

## Project Structure

- `main.py`: Core logic for DICOM decompression and conversion to 16-bit PNG files.
- `cleaner.py`: A Tkinter-based GUI tool to manually verify and sort images into `sorted` or `discarded` datasets.
- `labeler.py`: A custom OpenCV-based OBB (Oriented Bounding Box) labeling tool.
- `v12_train_setup.py`: Prepares the dataset by organizing labeled positive and sampled negative images into a YOLO-compatible structure.
- `prepare_yolo.py`: Handles DICOM-to-PNG conversion and final dataset splitting (train/val).
- `train_yolo.py`: Trains the YOLOv11/v12 OBB model using the Ultralytics framework.
- `test_model.py`: Runs inference with the trained model and sorts images into `detected` and `undetected` categories for manual review.
- `data.yaml`: Configuration file for the YOLO training process (paths and classes).

## Workflow

### 1. Preprocessing
Convert raw `.dcm` (DICOM) files from the `pos/` and `neg/` directories into high-bit-depth PNGs using `main.py`.

### 2. Data Cleaning
Use `cleaner.py` to filter through the converted PNGs, removing low-quality or irrelevant images.
- Images to keep are moved to `dataset_sorted/pos`.
- Images to discard are moved to `dataset_discarded/pos`.

### 3. Labeling
Run `labeler.py` to annotate the fabella in the sorted images. This tool generates OBB coordinates (class index, x1, y1, x2, y2, x3, y3, x4, y4) saved as `.txt` files in the `labels/pos/` directory.

### 4. Dataset Preparation
Run `v12_train_setup.py` or `prepare_yolo.py` to generate the `yolo_dataset/` directory. This script:
- Splites data into `train/` and `val/` sets.
- Pairs labeled positive images with background negative images (randomly sampled).
- Updates `data.yaml`.

### 5. Training
Execute `train_yolo.py` to begin training.
- Default resolution: `1024x1024` (to preserve bone texture).
- Default model: `yolo12n-obb.pt` (falls back to `yolo11n-obb` if unavailable).
- Checkpoints are saved in `runs/obb/`.

### 6. Testing & Inference
Use `test_model.py` to run the best-performing model (`runs/obb/fabella_obb_v12/weights/best.pt`) on new or unverified data. Results are sorted into `test_output/detected` and `test_output/undetected`.

## Requirements

- Python 3.10+
- `ultralytics` (YOLO)
- `pydicom` (DICOM handling)
- `opencv-python` (Image processing & UI)
- `numpy`
- `pillow`
- `torch` (CUDA recommended)
- `pylibjpeg` / `gdcm` (For compressed DICOM support)

## Setup

1. Install dependencies:
   ```bash
   pip install ultralytics pydicom opencv-python numpy pillow torch
   ```
2. Place raw DICOM files in `pos/` (for fabella cases) and `neg/` (for healthy cases).
3. Follow the workflow steps listed above.
