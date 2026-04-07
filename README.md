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
├── sorter.py               # Image sorting/review UI with classifier triage
├── labeler_obb.py          # Oriented Bounding Box annotation
├── labeler_seg.py          # Polygon segmentation annotation
├── labeler_sam.py          # SAM3/RF-DETR model-assisted auto-labeling
├── augmentation.py         # Augmentation engine (OpenCV/NumPy)
├── prepare_dialog.py       # Augmentation config modal UI
├── preparer.py             # Dataset preparation + COCO export
├── trainer.py              # Unified multi-architecture trainer
├── tester.py               # Unified multi-architecture tester
├── classifier_utils.py     # Torchvision classifier pipeline utilities
├── shape_analysis.py       # 2D contour shape analysis tab
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
| RF-DETR | rfdetr | COCO | n, s, m, l |
| RF-DETR Seg | rfdetr | COCO | n, s, m, l, xl, 2xl |
| Torchvision Classifier | torchvision | sorted_dirs | efficientnet_v2_s, resnet50, resnet18, mobilenet_v3_small |

After training completes, a results chart (loss + val metrics over epochs) is displayed automatically. Classifier runs also save `history.json` with per-epoch metrics.

> **Note:** The Torchvision Classifier trains directly from `data/sorted/pos` and `data/sorted/neg` — no dataset prepare step is needed.

### 3. Model Testing

Click **Test** to run inference on unsorted images.
- Detection/segmentation models: results saved to `output/test_{type}/detected/` and `undetected/`.
- Classifier models: results saved to `output/test_{type}/auto_positive/`, `review_band/`, and `remaining_manual/`, with score labels drawn on each image.

### 4. Shape Analysis

The **Shape Analysis** tab loads YOLO seg labels from `data/labels/seg` and computes 2D contour metrics (area, perimeter, circularity, solidity, aspect ratio, elongation, equivalent diameter). Results can be visualised as scatter plots or histograms and exported to CSV.

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
`SegLabeler` — polygon segmentation annotation tool. Click to place vertices, right-click to close. Saves labels in YOLO segment format. Automatically mirrors each image into sibling `pos_labeled/` / `pos_unlabeled/` directories as labels are added or removed.

### `labeler_sam.py`
`SAM3AutoLabeler` — model-assisted auto-labeling tool. Supports SAM3 segment-everything or a trained RF-DETR Seg model as the proposal backend. References are used to rank candidates; the review loop is identical across backends. Images are mirrored into sibling `pos_labeled/` / `pos_unlabeled/` directories as labels are saved.

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
`ModelTrainer` — unified trainer with `ARCHITECTURES` registry. Dispatches to ultralytics, RF-DETR, or torchvision backends. Torchvision Classifier training uses class-weighted cross-entropy, AdamW, ReduceLROnPlateau, and F1-based early stopping. Generates `best_classifier.pth`, `last_classifier.pth`, `results.png`, and `history.json`. Includes `ModelRegistry` for tracking trained models across all architectures.

### `tester.py`
`ModelTester` — architecture-aware inference runner. Detection/segmentation models route results to `detected/` or `undetected/`. Classifier models score each image and route to `auto_positive/`, `review_band/`, or `remaining_manual/` with annotated score overlays.

### `classifier_utils.py`
Shared utilities for the Torchvision Classifier pipeline: backbone registry (EfficientNet V2 S, ResNet50, ResNet18, MobileNet V3 Small), `ImagePathDataset`, train/val transform builders, checkpoint I/O, `predict_fabella_probability`, and `load_png_bgr_for_overlay`.

### `shape_analysis.py`
`ShapeAnalysisTab` — Tkinter tab embedded in the main notebook. Loads YOLO seg polygon labels, computes 2D shape metrics (area, perimeter, circularity, convexity, solidity, aspect ratio, elongation, equivalent diameter), and displays scatter plots and histograms via embedded Matplotlib. Supports CSV export.

---

## Version Log

### `testing` branch

#### v0.5.0 — Torchvision Classifier, Shape Analysis, RF-DETR Auto-Label Backend & UI Fixes

**New modules**
- `classifier_utils.py` — complete binary classification pipeline: backbone registry (EfficientNet V2 S, ResNet50, ResNet18, MobileNet V3 Small), `ImagePathDataset`, augmented train / center-crop val transforms, checkpoint save/load (stores backbone key, imgsz, and triage thresholds), `predict_fabella_probability`, and `load_png_bgr_for_overlay`
- `shape_analysis.py` — `ShapeAnalysisTab`: Tkinter tab with embedded Matplotlib scatter/histogram plots and summary statistics table for 2D contour metrics computed from YOLO seg labels; CSV export; background threading

**Training (`trainer.py`)**
- New `Torchvision Classifier` architecture in `ARCHITECTURES` registry (backend `torchvision`, task `classify`, sizes `efficientnet_v2_s` / `resnet50` / `resnet18` / `mobilenet_v3_small`)
- Full `_train_torchvision_classifier` loop: class-weighted `CrossEntropyLoss`, AdamW (lr=1e-4), `ReduceLROnPlateau` on val F1 (patience=3), F1-based early stopping (patience=10), saves `best_classifier.pth` + `last_classifier.pth`, dual-panel `results.png` at 150 dpi, `history.json`
- `ModelRegistry` now resolves `best_classifier.pth` for classifier runs; `_infer_from_name` recognises `torchvision_classifier_*` prefixes
- RF-DETR trainer uses native model resolution from `model_config` instead of rounding UI imgsz to nearest 32; extracted `RFDETR_SEG_TRAIN_KWARGS` and `RFDETR_SEG_LOSS_KWARGS` constants; fixed `xyxy_to_xywh` supervision patch to use in-place copy
- RF-DETR Seg size tokens updated to `n/s/m/l/xl/2xl`

**Testing (`tester.py`)**
- Classifier test path creates three subdirs: `auto_positive/`, `review_band/`, `remaining_manual/`
- Score + band label drawn on each output image with colour coding (green / orange / grey)
- Reads stored thresholds from checkpoint metadata

**Sorting UI (`sorter.py`)**
- New `ClassifierTriageDialog`: model picker from `ModelRegistry`, configurable auto-positive and review thresholds with validation
- `MODEL TRIAGE` button in `FabellaCleaner`: runs selected classifier over remaining images, auto-moves high-confidence positives to `sorted/pos`, queues review-band images first
- Per-image model score label with colour-coded band display (AUTO-POSITIVE / REVIEW BAND / MANUAL)

**Main app (`app.py`)**
- `Shape Analysis` tab added to the main notebook
- `AutoLabelerConfigDialog`: selects SAM3 or RF-DETR Seg backend and confidence threshold before launching the auto-labeler
- UI thread-safety: all `root.after(0, ...)` calls replaced with a `queue.Queue` + 50 ms pump (`_enqueue_ui` / `_drain_ui_queue`), with graceful `TclError` suppression on window close
- Size combobox widened to 20; `_on_size_change` auto-sets imgsz to the backbone's recommended default for classifier architectures
- Prepare dataset blocked with an info message for classifier arch (no YOLO prepare needed)

**Auto-labeler (`labeler_sam.py`)**
- Dual proposal backends: SAM3 segment-everything (unchanged) or a trained RF-DETR Seg model, selectable at launch
- `labeled_dir` / `unlabeled_dir` sibling folders auto-created and backfilled on init; updated on every label save
- `_has_label` now checks file size > 0; `_draw_progress` replaced with console-only output to avoid corrupting the review OpenCV window
- OpenCV review window opened after batch prediction completes (not before)
- HUD displays active backend and threshold per frame

**Seg labeler (`labeler_seg.py`)**
- `labeled_dir` / `unlabeled_dir` sync logic (same as SAM labeler): backfilled on init, updated on save and navigation
- `has_saved_label` checks size > 0

**Augmentation & preparer fixes**
- `_transform_labels_resize`: new function to remap normalised polygon coordinates through the same Stretch / Fit-Pad / Crop transform applied to the image
- `apply_preprocessing_with_labels`: new public API pairing image and label transforms
- `apply_preprocessing`: default width/height now reads from `img.shape` instead of hardcoded 1024
- `_to_uint8`: fixed `cv2.normalize()` to pass an explicit `np.empty` dst buffer (avoids crash in newer OpenCV)
- `YoloPreparer._process_single`: uses `apply_preprocessing_with_labels` so label coordinates stay aligned after resize

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
