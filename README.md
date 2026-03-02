# FabellaFilter

A complete end-to-end medical imaging pipeline for detecting the fabella bone in X-ray images. Covers DICOM conversion, image sorting, polygon/OBB labeling, dataset preparation with augmentation, and multi-architecture model training and testing — all through a single Tkinter desktop UI.

---

## Features

- **DICOM → PNG conversion** with GPU acceleration (PyTorch)
- **Manual image sorting** UI (keep / discard / label positive vs negative)
- **Polygon segmentation labeler** and **Oriented Bounding Box (OBB) labeler**
- **Dataset generator** with configurable augmentation pipeline (flip, rotate, crop, shear, brightness, blur, noise, mosaic, and more)
- **Multi-architecture training**: YOLO Seg, YOLO OBB, RT-DETR, RF-DETR
- **COCO export** for RF-DETR (Roboflow layout: `train/`, `valid/`, `test/`)
- **Training results chart** auto-displayed after each run (matplotlib)
- **CSV dataset manifest** for reproducibility and audit trail
- Real-time log output and progress bars in UI

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (tested on RTX 4070, CUDA 12.4)

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python pydicom pillow matplotlib numpy
pip install -U transformers peft
pip install -U rfdetr
```

> **Note:** RF-DETR requires `transformers` and `peft` to be upgraded first before installing `rfdetr`.

---

## Project Structure

```
FabellaFilter-main/
├── main_app.py             # Main UI entry point
├── dicom_converter.py      # DICOM → PNG conversion
├── dataset_cleaner.py      # Manual image sorting UI
├── obb_labeler.py          # Oriented Bounding Box labeler
├── seg_labeler.py          # Polygon segmentation labeler
├── yolo_preparer.py        # Dataset preparation + COCO export
├── dataset_generator.py    # Augmentation config modal
├── augmentor.py            # Augmentation engine (OpenCV/NumPy)
├── model_trainer.py        # Unified multi-arch trainer
├── model_tester.py         # Unified multi-arch tester
├── data/
│   ├── raw/                # Raw DICOM files (pos/ and neg/)
│   ├── png/                # Converted PNG images
│   ├── sorted/             # Manually reviewed images (pos/ and neg/)
│   ├── discarded/          # Discarded images
│   ├── labels/             # Annotation files (obb/ and seg/)
│   ├── yolo/               # YOLO dataset splits + YAML
│   └── coco/               # COCO format dataset (for RF-DETR)
└── output/
    ├── runs/               # Training run outputs
    └── test_*/             # Test inference results
```

---

## Usage

```bash
python main_app.py
```

This opens the main dashboard. Follow the pipeline tabs in order:

### 1. Dataset Tools

| Step | Tool | Description |
|------|------|-------------|
| 1 | **Convert DICOM** | Converts `.dcm` files from `data/raw/pos` and `data/raw/neg` to 16-bit PNG |
| 2 | **Clean Dataset** | Opens sorting UI — arrow keys to navigate, space to keep, delete to discard |
| 3 | **Sort Negatives** | Same UI for negative images — route to `sorted/pos` (has fabella) or `sorted/neg` |
| 4 | **Label OBB** | Click 3 points per image to draw oriented bounding boxes |
| 5 | **Label Seg** | Draw polygon masks for segmentation training |
| 6 | **Prepare Dataset** | Configure augmentation, splits, image size, then generate the training dataset |

### 2. Model Training

Select architecture, size, epochs, batch size, and image size, then click **Train**.

| Architecture | Backend | Format | Sizes |
|---|---|---|---|
| YOLO Seg | ultralytics | YOLO | n, s, m, l, x |
| YOLO OBB | ultralytics | YOLO | n, s, m, l, x |
| RT-DETR | ultralytics | YOLO | l, x |
| RF-DETR | rfdetr | COCO | base, large |

After training completes, a results chart (loss + class error over epochs) is displayed automatically.

### 3. Model Testing

Click **Test** to run inference on unsorted images. Results are saved to `output/test_{arch}_{size}/detected/` and `undetected/`.

---

## Modules

### `main_app.py`
Entry point. Tkinter window with **Dataset Tools** and **Model Training** tabs. Manages threading, progress windows (`PrepareProgressWindow`), log output, and post-training chart popup.

### `dicom_converter.py`
`DicomConverter` — converts raw `.dcm` files to 16-bit PNG. Handles windowing, rescaling, photometric interpretation, and GPU-accelerated batch processing via PyTorch.

### `dataset_cleaner.py`
`FabellaCleaner` — Tkinter `Toplevel` sorting UI. Supports zoom, pan, keyboard navigation, and undo (Ctrl+Z). Configurable routing for positive/negative sorting workflows.

### `obb_labeler.py`
`OBBLabeler` — OpenCV tool for Oriented Bounding Box annotation. Click 3 points to define a rotated rectangle. Saves labels in YOLO OBB format.

### `seg_labeler.py`
`SegLabeler` — polygon segmentation annotation tool. Click to place vertices, right-click to close. Saves labels in YOLO segment format.

### `yolo_preparer.py`
`YoloPreparer` — full dataset preparation pipeline:
- Prioritises reviewed `data/sorted/neg` negatives; falls back to raw DICOM conversion with a warning
- 80/20 train/val split with configurable augmentation multiplier
- Writes YOLO YAML config with forward-slash paths (ultralytics compatible)
- Exports CSV manifest with per-sample metadata
- Exports COCO JSON (`train/`, `valid/`, `test/` with `_annotations.coco.json`) for RF-DETR
- `step_callback(pct, label)` for progress bar integration

### `dataset_generator.py`
`DatasetGeneratorModal` — augmentation configuration modal. Scrollable UI with per-augmentation toggles and parameter sliders. Returns config dict consumed by `yolo_preparer.py`.

### `augmentor.py`
Pure OpenCV/NumPy augmentation engine. Functions: `read_labels`, `write_labels`, `pair_points`, `flat_points`, `apply_preprocessing`, `augment_sample`. Supports: flip, 90° rotate, arbitrary rotation, crop, shear, brightness, exposure, saturation, hue, blur, noise, mosaic.

### `model_trainer.py`
`ModelTrainer` — architecture-agnostic unified trainer. `ARCHITECTURES` registry maps each arch to backend, task, format, and size/version options. Dispatches to `_train_ultralytics()` or `_train_rfdetr()`. RF-DETR training captures stdout to parse per-epoch metrics and generates a `results.png` chart via matplotlib. Patches `supervision.xyxy_to_xywh` compatibility for supervision ≥ 0.26.

### `model_tester.py`
`ModelTester` — architecture-aware inference runner. Loads weights from `output/runs/{run_name}/weights/best.pt`. Draws OBB polygons, segmentation masks, or detection boxes on test images and routes results to `detected/` or `undetected/` folders.

---

## Version Log

### `testing` branch

#### v0.3.1 — Cleanup
- Deleted `yolo_trainer.py` (superseded by `model_trainer.py`)
- Deleted `yolo_tester.py` (superseded by `model_tester.py`)
- Deleted `__init__.py` (empty leftover package marker)
- Removed dead `sys.path.append` and unused `import sys` from `main_app.py`

#### v0.3.0 — Multi-Architecture Support & RF-DETR Pipeline
- **New**: `model_trainer.py` — unified `ModelTrainer` with `ARCHITECTURES` registry (YOLO Seg, YOLO OBB, RT-DETR, RF-DETR)
- **New**: `model_tester.py` — architecture-aware `ModelTester` replacing old YOLO-only tester
- **COCO export** in `yolo_preparer.py` — Roboflow-layout `train/`, `valid/`, `test/` splits with `_annotations.coco.json`; auto-migrates old `val/` folder naming
- **Detection task** support: `_poly_to_bbox()` converts polygon labels to `[cx, cy, w, h]` for RT-DETR
- **Sorted negatives priority**: preparer checks `data/sorted/neg` first, falls back to raw DICOM
- **`PrepareProgressWindow`** modal with determinate progress bar and scrollable log
- **Training results chart**: matplotlib 2-panel plot (loss + class error) auto-popup after training
- **RF-DETR compatibility fixes**: `val→valid` rename, `test/` mirror, `supervision.xyxy_to_xywh` monkey-patch, stdout capture for metric parsing
- **Noise suppression**: `TF_ENABLE_ONEDNN_OPTS`, `TF_CPP_MIN_LOG_LEVEL`, `NO_ALBUMENTATIONS_UPDATE` set before any imports
- **`ScrollableFrame` mousewheel fix**: scoped to hover (`bind('<Enter>'>`/`bind('<Leave>')`) to prevent crash after modal destroy
- **Path consistency**: all paths use `os.path.join()` and `.replace(os.sep, '/')` for YAML/log output
- **Model tab UI**: Architecture, Version, Size, Epochs, Batch, ImgSize controls with dynamic updates per arch

---

### `main` branch

#### v0.2.0 — Augmentation Pipeline, Seg Labeler, Neg Sorter
- Dataset augmentation modal with full OpenCV/NumPy engine (`augmentor.py`)
- Segmentation polygon labeler (`seg_labeler.py`)
- Negative image sorter routing to `sorted/pos` or `sorted/neg`
- CSV manifest saved alongside YAML for every dataset prepare run
- YOLO versions v8 through v26 supported

#### v0.1.2 — GPU DICOM Conversion
- PyTorch GPU acceleration for DICOM → PNG batch conversion

#### v0.1.1 — DICOM Metadata Export
- Demographic and study data extracted from DICOM headers to CSV

#### v0.1.0 — Initial Modular Refactor
- Monolithic script split into modular components
- Tkinter UI with Dataset Tools and Model Training tabs
- Threading for non-blocking UI during long operations

---

## License

See [LICENSE](LICENSE).
