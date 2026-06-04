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

**In-canvas display menu** (press `M` or click the MENU badge, top-right):
- **Colour swatches** — 8 preset fill colours (Green, Yellow, Cyan, Magenta, Red, Orange, Blue, White); visual only, no effect on saved label files
- **Opacity cycle** — cycles fill transparency from 0 % (outline only) to 50 % in 10 % steps; also bound to `T`
- **Direction markers** — toggled via the menu or `V`; places numbered vertex markers at evenly-spaced intervals (e.g. vertices 1, 6, 11, 16, 21 for a 25-point polygon) and shows a CW / CCW winding label above the shape

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

#### v0.5.6 — Next-Gen NumPy/SciPy Algorithmic Optimizations

**Vectorized Contour Resampling & Feature Extraction ([shape_analysis.py](shape_analysis.py) & [labeler_sam.py](labeler_sam.py))**
- Rewrote contour resampling (`resample_closed_contour`) from a slow iterative `while` loop to a fully vectorized 1D linear interpolation using `np.interp` and `np.column_stack`.
- Vectorized the entire shoelace area calculation, bounding box metrics, and perimeter calculations in the active learning auto-labeler (`_get_features`), switching to native NumPy rolls and vector operations.
- Replaced the high-memory pairwise distance tensor-broadcasting calculation `points[:, None, :] - points[None, :, :]` in `pairwise_max_distance` with an extremely efficient calculation using SciPy's vectorized `pdist` distance estimator.

**Fast Index-Based PERMANOVA Permutation ([shape_analysis.py](shape_analysis.py))**
- Optimized the computational bottleneck in Euclidean PERMANOVA calculations. Replaced slow string-matching labels inside the permutation loop with pre-computed index mapping (`label_to_indices`).
- Substituted heavy string label array permutations with mathematically equivalent index shuffles (`rng.shuffle(indices)`), boosting permutation math throughput by an order of magnitude.
- Calculated the total sum-of-squares once to compute `observed_within` via `total - observed_between`, eliminating unnecessary redundant calculations.

#### v0.5.5 — High-Performance LineCollection Plotting & Scan Completed Warnings

**Vectorized Contour Plotting ([shape_analysis.py](shape_analysis.py))**
- Replaced slow, iterative Matplotlib `.plot()` calls inside `_render_alignment_figure` with vectorized `matplotlib.collections.LineCollection` arrays.
- Eliminates multi-minute UI freeze and "Not Responding" thread locks when parsing large clinical cohorts (e.g., over 1,300 complex contours), converting rendering transactions into near-instantaneous canvas draws.
- Manually bounds axes limits with 5% outer safety padding margins to properly auto-scale Procrustes alignments inside both comparative grids.

**Cohort Generation Warnings ([shape_analysis.py](shape_analysis.py))**
- Intercepted background analysis callback routines inside `_on_analysis_complete` to trigger a clean informational dialog modal (`messagebox.showinfo`) upon successful cohort scans.
- Informs the user that the database compilation has completed and that the program is now preparing demographics matrices, PCA grids, and scree diagrams, instructing them to pause interactions while plots finalise.

#### v0.5.4 — Cross-Platform Compatibility & Changeable Shape-Analysis Metadata Root

**Cross-Platform Paths ([shape_analysis.py](shape_analysis.py))**
- Resolved path failures occurring when moving the application from Windows to Linux/Ubuntu by decoupling the hardcoded Windows path `E:\Emory`.
- Configured a local platform-agnostic fallback to the workspace's `data/` directory using `os.path.normpath` and `os.path.join`, ensuring automatic error-free execution on non-Windows operating systems.

**Interactive Path Selection ([shape_analysis.py](shape_analysis.py))**
- Integrated an intuitive **Folder Paths Configuration** card directly within the **Shape Analysis** tab, featuring a styled entry box bound to `self.emory_root_var` and a "Browse..." button mapped to Tkinter's `filedialog.askdirectory`.
- Added manual `<Return>` entry binding to let users easily type arbitrary paths, hit Enter, and trigger instant, threaded rescans of the new demographic/clinical databases.
- Disabled the shape-analysis computation from starting automatically on application load, allowing users to configure or inspect their targets before manual execution via the **Refresh / Rescan** panel.

#### v0.5.3 — Thread-Safe Background Auto-Labeler with Dynamic Compute Throttling

**Non-Blocking Active Predictor Window ([labeler_sam.py](labeler_sam.py))**
- Replaced the blocking synchronous batch segmentation pipeline with a modern, non-blocking Thread/Modal progress dialog architecture. Background-threaded processing keeps the application 100% interactive and avoids OS "Not Responding" or lock-up messages.
- Added mouse pointer tracking using `winfo_pointerxy()` to assess system-wide computer usage. When active interaction is detected, prediction introduces brief custom sleep delays to yield hardware resources to user applications, restoring maximum compute speed once the user is idle.
- Implemented real-time status logging, status color badges (Idle/Active), a **Pause/Resume Model** toggle button to suspend and restart model predictions at will, and a **Cancel / Exit** process button.

**Compute Optimization Control ([app.py](app.py))**
- Integrated dynamic compute optimization settings inside the **Auto-Label Model Setup** configuration dialog, featuring full-theme support, a dynamic bypass checkbutton, and customized float inputs for active processing delays and idle timeouts.
- Configured safe callback handlers to pass parent bindings and dynamic throttle variables into the SAM3 and RF-DETR auto-labelers.

#### v0.5.2 — Dynamic Folder Path Configuration & Selection

**Folder Configuration UI (`app.py`)**
- Added a brand-new **Folder Paths Configuration** panel directly within the **Dataset Tools** tab in the main UI, featuring 2-column grid alignment, styling consistent with VS Code Dark Theme, and "Browse..." buttons connected to Tkinter `filedialog.askdirectory`.
- Replaced all hardcoded string paths in data-scanning modules (Convert DICOM, Clean Dataset, Sort Negatives, Label OBB, Label Segmentation, SAM3 Auto-Label, Dataset Preparation, and Model Testing) with dynamic Tkinter `tk.StringVar` references (`self.raw_dir_var`, `self.png_dir_var`, `self.sorted_dir_var`, `self.discarded_dir_var`, `self.obb_label_dir_var`, and `self.seg_label_dir_var`).
- Integrated double-direction dynamic trace bindings so editing or browsing a folder path immediately triggers `_refresh_dataset_stats()` and refreshes the **Dataset Overview** overview panel in real-time.

**Flexible Tool Integration**
- **DICOM Converter (`converter.py` / `app.py`)** — instantiates with custom source raw folder and target PNG folder.
- **Dataset Cleaner / Negative Sorter (`sorter.py` / `app.py`)** — instantiates `FabellaCleaner` with custom source/keep/discard target directories under the hood.
- **OBB, Segmentation & Active SAM Labelers (`labeler_obb.py` / `labeler_seg.py` / `labeler_sam.py` / `app.py`)** — custom image and label folders are passed to constructors and resolved cleanly.
- **Dataset Preparation (`preparer.py` / `prepare_dialog.py` / `app.py`)** — updated `YoloPreparer` and `DatasetGeneratorModal` to accept customizable image and label directories, establishing clean relative paths dynamically.
- **Model Training & Testing (`trainer.py` / `tester.py` / `app.py`)** — updated `ModelTrainer` to pass customized positive/negative sorted validation parameters to `gather_sorted_samples`, and updated `ModelTester` to use live image and sorted output folder parameters.

#### v0.5.1 — Seg Labeler Display Menu

**Seg labeler (`labeler_seg.py`)**
- In-canvas floating menu panel (toggle `M` or click MENU badge, top-right) with three controls:
  - **Colour swatches** — 8 preset polygon fill colours; purely visual, no effect on saved labels
  - **Opacity cycle** — fill transparency 0 %→10 %→…→50 %; also bound to `T`
  - **Direction markers** — toggle with menu button or `V`; marks up to 5 evenly-spaced polygon vertices with their actual 1-based index (e.g. 1 / 6 / 11 / 16 / 21 for a 25-point polygon) and renders a CW / CCW winding label above the shape
- Polygon fill/outline colour now derived from `self.poly_color`; outline auto-darkened at 65 %
- Fill overlay skipped entirely when opacity is 0 %
- Hint text updated to include `M: Menu | T: Opacity | V: Direction`

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
