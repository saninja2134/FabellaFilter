# Fabella Dataset Manager UI

This directory contains the refactored, object-oriented, and unified user interface for the Fabella Dataset Manager.

## Overview

The application has been broken down into modular components, each responsible for a specific part of the dataset processing and model training pipeline. This ensures separation of concerns and makes the codebase easier to maintain and extend.

## Modules

### `main_app.py`
The entry point for the unified application. It creates the main Tkinter window with tabs for "Dataset Tools" and "Model Training". It imports and utilizes the other modules to perform actions when buttons are clicked. It also includes a logging area to display output from the various processes directly in the UI.

### `dicom_converter.py`
Contains the `DicomConverter` class.
- **Purpose:** Converts raw DICOM files (`.dcm`) into 16-bit PNG images suitable for processing and labeling.
- **Key Features:** Handles windowing, rescaling, and photometric interpretation to ensure bones are visible and correctly oriented (white bones on black background).

### `dataset_cleaner.py`
Contains the `FabellaCleaner` class.
- **Purpose:** Provides a UI (as a `tk.Toplevel` window) to manually sort converted PNG images into "Keep" (`dataset_sorted`) or "Discard" (`dataset_discarded`) folders.
- **Key Features:** Supports zooming, panning, keyboard shortcuts (Left/Right arrows), and undo functionality (Ctrl+Z).

### `obb_labeler.py`
Contains the `OBBLabeler` class.
- **Purpose:** An OpenCV-based tool for annotating images with Oriented Bounding Boxes (OBB).
- **Key Features:** Allows users to click 3 points to define an oriented rectangle. Supports zooming (scroll wheel), panning (middle mouse drag), and saving labels in YOLO OBB format.

### `yolo_preparer.py`
Contains the `YoloPreparer` class.
- **Purpose:** Prepares the dataset for YOLO training.
- **Key Features:** Gathers labeled positive images, samples an equal number of negative (background) images, splits them into training and validation sets (80/20 split), and generates the required `data.yaml` configuration file.

### `yolo_trainer.py`
Contains the `YoloTrainer` class.
- **Purpose:** Handles the training of the YOLO model.
- **Key Features:** Checks for GPU availability, loads the specified YOLO model (e.g., `yolo12n-obb.pt` or falls back to `yolo11n-obb.pt`), and initiates the training process with predefined hyperparameters suitable for medical images.

### `yolo_tester.py`
Contains the `YoloTester` class.
- **Purpose:** Runs inference using the trained YOLO model on a set of test images.
- **Key Features:** Loads the best trained weights, runs predictions on unsorted images, draws the detected bounding boxes and confidence scores on the images, and saves the results into "detected" and "undetected" folders for review.

## Usage

To run the unified application, execute the `main_app.py` script:

```bash
python UI/main_app.py
```

This will open the main dashboard where you can access all the tools in the pipeline sequentially.
