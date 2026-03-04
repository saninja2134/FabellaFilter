# FabellaFilter

End-to-end medical imaging pipeline for detecting the **fabella** (sesamoid bone) in lateral knee X-rays. Covers DICOM conversion, image sorting, annotation, dataset preparation with augmentation, and multi-architecture model training & testing — all from a single desktop UI.

---

## Features

- **DICOM → PNG conversion** — batch convert raw DICOM files to 16-bit PNG
- **Image sorting UI** — keyboard-driven review to keep, discard, or route images
- **Annotation tools** — polygon segmentation labeler, OBB labeler, and SAM-assisted auto-labeling
- **Dataset preparation** — configurable augmentation pipeline (flip, rotate, crop, shear, brightness, blur, noise, mosaic, etc.), 80/20 train/val split, JPEG compression
- **Multi-architecture training** — YOLO Seg, YOLO OBB, RT-DETR, RF-DETR (Nano through Large)
- **Dual export** — YOLO format + COCO JSON (Roboflow layout) generated simultaneously
- **Training results chart** — auto-displayed after each run
- **CSV dataset manifest** — per-sample metadata for reproducibility
- Real-time log output and progress bars in the UI

---

## Requirements

- **Python 3.10+** (3.12 recommended)
- **Conda** (Anaconda or Miniconda) — strongly recommended for environment management
- **CUDA-capable GPU** recommended (tested on RTX 4070, CUDA 12.4)

### Setup with Conda

```bash
# Create and activate environment
conda create -n fabella python=3.12 -y
conda activate fabella

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# Install remaining dependencies
pip install ultralytics opencv-python pydicom pillow matplotlib numpy scikit-learn
pip install -U transformers peft
pip install -U rfdetr
```

> **Note:** Install `transformers` and `peft` before `rfdetr` — RF-DETR depends on up-to-date versions of both.

---

## Project Structure

```
FabellaFilter/
├── app.py                  # Main UI — entry point
├── converter.py            # DICOM → PNG conversion
├── sorter.py               # Image sorting/review UI
├── labeler_obb.py          # Oriented Bounding Box annotation
├── labeler_seg.py          # Polygon segmentation annotation
├── labeler_sam.py          # SAM-assisted auto-labeling
├── augmentation.py         # Augmentation engine (OpenCV/NumPy)
├── prepare_dialog.py       # Augmentation config modal UI
├── preparer.py             # Dataset preparation + COCO export
├── trainer.py              # Unified multi-architecture trainer
├── tester.py               # Unified multi-architecture tester
├── data/
│   ├── raw/                # Raw DICOM files (pos/ and neg/)
│   ├── png/                # Converted PNG images
│   ├── sorted/             # Reviewed images (pos/ and neg/)
│   ├── labels/             # Annotations (obb/ and seg/)
│   ├── yolo/               # YOLO dataset splits + YAML
│   └── coco/               # COCO format dataset (for RF-DETR)
└── output/
    ├── runs/               # Training run outputs
    └── test_*/             # Test inference results
```

---

## Usage

```bash
conda activate fabella
python app.py
```

This opens the main dashboard. Follow the pipeline tabs in order:

### 1. Dataset Tools

| Step | Tool | Description |
|------|------|-------------|
| 1 | **Convert DICOM** | Converts `.dcm` files from `data/raw/` to 16-bit PNG |
| 2 | **Clean Dataset** | Sorting UI — arrow keys to navigate, space to keep, delete to discard |
| 3 | **Sort Negatives** | Route negative images to `sorted/pos` or `sorted/neg` |
| 4 | **Label OBB** | Click 3 points per image to draw oriented bounding boxes |
| 5 | **Label Seg** | Draw polygon masks for segmentation training |
| 6 | **SAM Auto-Label** | SAM-assisted auto-labeling for faster annotation |
| 7 | **Prepare Dataset** | Configure augmentation, splits, image size, then generate the training dataset |

### 2. Model Training

Select architecture, size, epochs, batch size, and image size, then click **Train**.

| Architecture | Backend | Format | Sizes |
|---|---|---|---|
| YOLO Seg | ultralytics | YOLO | n, s, m, l, x |
| YOLO OBB | ultralytics | YOLO | n, s, m, l, x |
| RT-DETR | ultralytics | YOLO | l, x |
| RF-DETR | rfdetr | COCO | nano, small, medium, base, large |

After training completes, a results chart (loss + class error over epochs) is displayed automatically.

### 3. Model Testing

Click **Test** to run inference on unsorted images. Results are saved to `output/test_{type}/detected/` and `undetected/`.

---

## Modules

### `app.py`
Entry point. Tkinter window with **Dataset Tools** and **Model Training** tabs. Manages threading, progress windows, log output, and post-training chart display.

### `converter.py`
`DicomConverter` — batch converts raw `.dcm` files to 16-bit PNG. Handles DICOM windowing, rescale slope/intercept, and photometric interpretation.

### `sorter.py`
`FabellaCleaner` — Tkinter sorting UI with zoom, pan, keyboard navigation, and undo (Ctrl+Z). Configurable routing for positive/negative sorting workflows.

### `labeler_obb.py`
`OBBLabeler` — OpenCV annotation tool. Click 3 points to define a rotated rectangle. Saves labels in YOLO OBB format.

### `labeler_seg.py`
`SegLabeler` — polygon segmentation annotation tool. Click to place vertices, right-click to close. Saves labels in YOLO segment format.

### `labeler_sam.py`
`SAM3AutoLabeler` — SAM-assisted auto-labeling tool for accelerated polygon annotation.

### `augmentation.py`
Pure OpenCV/NumPy augmentation engine. Supports flip, 90° rotate, arbitrary rotation, crop, shear, brightness, exposure, saturation, hue, blur, noise, and mosaic.

### `prepare_dialog.py`
`DatasetGeneratorModal` — augmentation configuration modal with per-augmentation toggles and parameter sliders. Returns config dict consumed by `preparer.py`.

### `preparer.py`
`YoloPreparer` — full dataset preparation pipeline:
- 80/20 train/val split with configurable augmentation
- JPEG output (quality 85) matching Roboflow compression
- YOLO YAML + COCO JSON (`train/`, `valid/`, `test/`) exported simultaneously
- CSV manifest with per-sample metadata
- Sorted negatives priority with raw DICOM fallback

### `trainer.py`
`ModelTrainer` — unified trainer with `ARCHITECTURES` registry. Dispatches to ultralytics or RF-DETR backends. RF-DETR training captures stdout for metric parsing and generates results charts. Includes `ModelRegistry` for tracking trained models.

### `tester.py`
`ModelTester` — architecture-aware inference runner. Draws OBB polygons, segmentation masks, or detection boxes on test images and routes results to `detected/` or `undetected/`.

---

## Version Log

### `testing` branch

#### v0.4.0 — Rename, Simplify & Compress
- **Renamed all modules** to clear, intuitive names (see Project Structure)
- **JPEG compression** for training images (quality 85, ~30-50 KB per 1024×1024 — matches Roboflow)
- **Codebase simplification** — 177 lines removed: deduplicated error handling, shared RF-DETR class maps, removed dead code and verbose comments, simplified DICOM conversion to CPU-only NumPy
- **Conda recommended** for environment setup

#### v0.3.1 — Cleanup
- Deleted superseded `yolo_trainer.py`, `yolo_tester.py`, `__init__.py`
- Removed dead imports

#### v0.3.0 — Multi-Architecture Support & RF-DETR Pipeline
- Unified `ModelTrainer` with `ARCHITECTURES` registry (YOLO Seg, YOLO OBB, RT-DETR, RF-DETR)
- Architecture-aware `ModelTester` replacing old YOLO-only tester
- COCO export — Roboflow-layout splits with `_annotations.coco.json`
- Detection task support via polygon-to-bbox conversion
- Sorted negatives priority with raw DICOM fallback
- Progress window with determinate progress bar and scrollable log
- Training results chart auto-popup
- RF-DETR compatibility fixes (val→valid, supervision monkey-patch, stdout capture)
- Noise suppression for TensorFlow and Albumentations warnings

### `main` branch

#### v0.2.0 — Augmentation Pipeline, Seg Labeler, Neg Sorter
- Dataset augmentation modal with full OpenCV/NumPy engine
- Segmentation polygon labeler
- Negative image sorter
- CSV manifest for every dataset prepare run

#### v0.1.2 — GPU DICOM Conversion
- PyTorch GPU acceleration for DICOM → PNG batch conversion

#### v0.1.1 — DICOM Metadata Export
- Demographic and study data extracted from DICOM headers to CSV

#### v0.1.0 — Initial Modular Refactor
- Monolithic script split into modular components
- Tkinter UI with Dataset Tools and Model Training tabs

---

## License

See [LICENSE](LICENSE).
