# Module for preparing the dataset for YOLO training.
import os
import shutil
import random
import csv
import json
import pydicom
import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split
import augmentor

class YoloPreparer:
    # A class to prepare and split the dataset for YOLO model training.
    def __init__(self, task="segment", pos_img_dir="data/sorted/pos", neg_dicom_dir="data/raw/neg"):
        # Initializes the YoloPreparer.
        # Args:
        # task (str): The task type ('segment', 'obb', or 'detect').
        # pos_img_dir (str): Directory containing positive images.
        # neg_dicom_dir (str): Directory containing negative DICOM images.
        self.task = task
        self.pos_img_dir = pos_img_dir
        # Detection uses segmentation polygon labels as source (bbox derived on write)
        if task == 'obb':
            task_key = 'obb'
        elif task == 'detect':
            task_key = 'det'
        else:
            task_key = 'seg'
        self.label_dir = os.path.join('data', 'labels', 'obb' if task == 'obb' else 'seg')
        self.neg_dicom_dir = neg_dicom_dir
        self.base_output = os.path.join('data', 'yolo', task_key)
        self.yaml_path = os.path.join('data', 'yolo', f'data_{task_key}.yaml')

    @staticmethod
    def _poly_to_bbox(coords):
        # Converts flat normalized polygon coords to YOLO detection bbox [cx, cy, w, h].
        pts = augmentor.pair_points(coords)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        return [cx, cy, x_max - x_min, y_max - y_min]

    def convert_dicom_to_16bit_png(self, src_path, dst_path, device=None):
        # Converts a single DICOM file to a 16-bit PNG.
        # 
        # Args:
        # src_path (str): Path to the source DICOM file.
        # dst_path (str): Path to save the converted PNG file.
        # device (torch.device, optional): Device to use for processing.
        #     
        # Returns:
        # bool: True if successful, False otherwise.
        try:
            ds = pydicom.dcmread(src_path)
            img = ds.pixel_array.astype(float)
            
            if device and device.type == "cuda":
                img_tensor = torch.from_numpy(img).to(device)
                
                if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
                    img_tensor = img_tensor * ds.RescaleSlope + ds.RescaleIntercept
                    
                if 'WindowCenter' in ds and 'WindowWidth' in ds:
                    wc = ds.WindowCenter
                    ww = ds.WindowWidth
                    if hasattr(wc, '__iter__'): wc = wc[0]
                    if hasattr(ww, '__iter__'): ww = ww[0]
                    img_min = float(wc) - float(ww) // 2
                    img_max = float(wc) + float(ww) // 2
                else:
                    img_flat = img_tensor.flatten()
                    img_min = torch.quantile(img_flat, 0.01).item()
                    img_max = torch.quantile(img_flat, 0.99).item()
                    
                img_tensor = torch.clamp(img_tensor, img_min, img_max)
                img_tensor = ((img_tensor - img_min) / (img_max - img_min) * 65535.0).to(torch.int32)
                
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img_tensor = 65535 - img_tensor
                    
                img = img_tensor.cpu().numpy().astype(np.uint16)
            else:
                if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                    
                if 'WindowCenter' in ds and 'WindowWidth' in ds:
                    wc = ds.WindowCenter
                    ww = ds.WindowWidth
                    if hasattr(wc, '__iter__'): wc = wc[0]
                    if hasattr(ww, '__iter__'): ww = ww[0]
                    img_min = float(wc) - float(ww) // 2
                    img_max = float(wc) + float(ww) // 2
                else:
                    img_min = np.percentile(img, 1)
                    img_max = np.percentile(img, 99)
                    
                img = np.clip(img, img_min, img_max)
                img = ((img - img_min) / (img_max - img_min) * 65535.0).astype(np.uint16)
                
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = 65535 - img
                
            cv2.imwrite(dst_path, img)
            return True
        except Exception as e:
            print(f"Error converting {src_path}: {e}")
            return False

    def setup_dataset(self, config=None, also_export_coco=True, progress_callback=None, step_callback=None):
        # Prepares the dataset by gathering positive samples, sampling negative samples,
        # splitting into train/val sets, and creating the data.yaml file.
        # COCO format is ALWAYS exported alongside YOLO (uses hard links, no space waste).
        #
        # Args:
        # config (dict, optional): Augmentation/preprocessing config from DatasetGeneratorModal.
        # also_export_coco (bool): Also write COCO JSON format (default True, always recommended).
        # progress_callback (callable): Called with a log string.
        # step_callback (callable): Called with (pct: int, label: str) for progress bar updates.
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

        def step(pct, label=""):
            if step_callback: step_callback(pct, label)

        multiplier = 1
        if config:
            multiplier = config.get('generation', {}).get('multiplier', 1)

        step(0, "Scanning labeled positives...")
        # 1. Gather all POS samples that have labels
        if not os.path.exists(self.label_dir):
            log(f"Error: {self.label_dir} does not exist.")
            return

        labeled_files = [f for f in os.listdir(self.label_dir) if f.endswith('.txt')]
        pos_samples = []

        if not os.path.exists(self.pos_img_dir):
            log(f"Error: {self.pos_img_dir} does not exist.")
            return

        for f in labeled_files:
            png_name = f.replace('.txt', '.png')
            png_path = os.path.join(self.pos_img_dir, png_name)
            lbl_path = os.path.join(self.label_dir, f)
            if os.path.exists(png_path):
                # Validate label has at least one annotation before including as positive
                labels_check = augmentor.read_labels(lbl_path)
                if labels_check:
                    pos_samples.append((png_path, lbl_path))
                else:
                    log(f"  Skipped {png_name}: label file is empty or invalid.")

        if not pos_samples:
            log("No labeled positive images found. Have you finished labeling?")
            return

        log(f"Found {len(pos_samples)} labeled positive images.")
        if config:
            log(f"Augmentation multiplier: x{multiplier}")
        step(5, "Positives loaded.")

        # 2. Gather NEG images
        # Priority: use sorted/reviewed PNGs from data/sorted/neg first.
        # Fall back to converting raw DICOMs only when no sorted negatives exist.
        sorted_neg_dir = "data/sorted/neg"
        sorted_neg_pngs = []
        if os.path.exists(sorted_neg_dir):
            sorted_neg_pngs = [f for f in os.listdir(sorted_neg_dir) if f.lower().endswith('.png')]

        neg_samples = []
        temp_neg_png = None

        if sorted_neg_pngs:
            # Use curated negatives that passed the Sort Negatives review step
            random.shuffle(sorted_neg_pngs)
            num_neg = min(len(sorted_neg_pngs), len(pos_samples))
            selected = sorted_neg_pngs[:num_neg]
            log(f"Using {num_neg} sorted (reviewed) negative images from {sorted_neg_dir}.")
            step(15, f"Loading {num_neg} sorted negatives...")
            for i, fname in enumerate(selected):
                neg_samples.append((os.path.join(sorted_neg_dir, fname), None))
                if (i + 1) % 20 == 0 or (i + 1) == num_neg:
                    step(15 + int(55 * (i + 1) / num_neg), f"Loading negatives {i+1}/{num_neg}...")
        else:
            # No sorted negatives yet — fall back to converting raw DICOMs
            log("No sorted negatives found. Falling back to raw DICOM conversion from data/raw/neg.")
            log("Tip: Run 'Sort Negatives' first to use reviewed background images.")

            if not os.path.exists(self.neg_dicom_dir):
                log(f"Error: {self.neg_dicom_dir} does not exist either. Cannot find any negatives.")
                return

            neg_dicoms = [f for f in os.listdir(self.neg_dicom_dir) if f.lower().endswith('.dcm')]
            random.shuffle(neg_dicoms)
            num_neg = min(len(neg_dicoms), len(pos_samples))
            selected_neg = neg_dicoms[:num_neg]

            log(f"Converting {num_neg} negative DICOM images...")
            step(15, f"Converting {num_neg} DICOM negatives...")

            temp_neg_png = "temp_neg_png"
            if not os.path.exists(temp_neg_png):
                os.makedirs(temp_neg_png)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for i, dcm in enumerate(selected_neg):
                png_name = dcm.replace('.dcm', '.png')
                dst_path = os.path.join(temp_neg_png, png_name)
                if self.convert_dicom_to_16bit_png(os.path.join(self.neg_dicom_dir, dcm), dst_path, device=device):
                    neg_samples.append((dst_path, None))
                if (i + 1) % 10 == 0 or (i + 1) == num_neg:
                    log(f"Converted {i + 1}/{num_neg} negative images...")
                    step(15 + int(55 * (i + 1) / num_neg), f"Converting DICOM {i+1}/{num_neg}...")

        # 3. Split into Train/Val
        all_samples = pos_samples + neg_samples
        train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)

        # 4. Build YOLO output structure
        if os.path.exists(self.base_output): shutil.rmtree(self.base_output)

        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.base_output, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.base_output, split, 'labels'), exist_ok=True)

        # Pre-load mosaic pool if mosaic augmentation is enabled
        mosaic_pool = None
        if config and config.get('augmentations', {}).get('mosaic', {}).get('enabled'):
            log("Pre-loading images for mosaic augmentation...")
            mosaic_pool = []
            for img_path, lbl_path in pos_samples:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    labels = augmentor.read_labels(lbl_path)
                    mosaic_pool.append((img, labels))

        # Accumulate CSV rows: [split, image_file, label_file, class_id, coords, augmented, original_source]
        csv_rows = []

        def write_sample_files(img, labels, split, fname, original_source, is_augmented):
            # Apply preprocessing, write image + label files, and record CSV row.
            if img is None:
                return
            if config:
                img = augmentor.apply_preprocessing(img, config)
            img_dst = os.path.join(self.base_output, split, 'images', fname)
            cv2.imwrite(img_dst, img)
            rel_img = os.path.join(split, 'images', fname)
            if labels:
                stem = os.path.splitext(fname)[0]
                lbl_fname = stem + '.txt'
                lbl_dst = os.path.join(self.base_output, split, 'labels', lbl_fname)
                # For detection task, convert polygon coords → YOLO bbox format
                if self.task == 'detect':
                    det_labels = [(cid, self._poly_to_bbox(coords)) for cid, coords in labels]
                    augmentor.write_labels(lbl_dst, det_labels)
                else:
                    augmentor.write_labels(lbl_dst, labels)
                rel_lbl = os.path.join(split, 'labels', lbl_fname)
                for cid, coords in labels:
                    coords_str = ' '.join(f'{c:.6f}' for c in coords)
                    csv_rows.append([split, rel_img, rel_lbl, cid, coords_str,
                                     is_augmented, original_source])
            else:
                # Negative sample — no label
                csv_rows.append([split, rel_img, '', '', '', is_augmented, original_source])

        def process_samples(samples, split):
            for img_path, lbl_path in samples:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                # Load image and its matching labels together so transforms stay in sync
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                labels = augmentor.read_labels(lbl_path) if lbl_path else None

                # Original copy (no augmentation)
                write_sample_files(img, labels, split, stem + '.png',
                                   original_source=os.path.basename(img_path),
                                   is_augmented=False)

                # Augmented copies (positive samples only, when multiplier > 1)
                if lbl_path and config and multiplier > 1:
                    for k in range(1, multiplier):
                        # Re-read originals so each augmentation is independent
                        aug_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        aug_labels = augmentor.read_labels(lbl_path)
                        # Preprocessing is applied inside write_sample_files; augment first
                        aug_img, aug_labels = augmentor.augment_sample(
                            aug_img, aug_labels, config, mosaic_pool)
                        aug_fname = f"{stem}_aug{k}.png"
                        write_sample_files(aug_img, aug_labels, split, aug_fname,
                                           original_source=os.path.basename(img_path),
                                           is_augmented=True)

        log("Building training set...")
        step(72, "Building training set...")
        process_samples(train_samples, 'train')
        log("Building validation set...")
        step(85, "Building validation set...")
        process_samples(val_samples, 'val')

        # CLEANUP: Remove temporary negative PNG conversions if we created them
        if temp_neg_png and os.path.exists(temp_neg_png):
            shutil.rmtree(temp_neg_png)
            log("Cleaned up temporary negative PNGs.")

        step(90, "Writing YAML config...")
        # 5. Create data.yaml
        yaml_content = f"""path: {os.path.abspath(self.base_output).replace(os.sep, '/')}
train: train/images
val: val/images

names:
  0: fabella
"""
        with open(self.yaml_path, 'w') as f:
            f.write(yaml_content)

        # 6. Write CSV manifest
        step(94, "Writing CSV manifest...")
        csv_path = os.path.join(self.base_output, 'dataset.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['split', 'image_file', 'label_file', 'class_id',
                             'coords', 'augmented', 'original_source'])
            writer.writerows(csv_rows)

        aug_pos = len([s for s in train_samples + val_samples if s[1]]) * max(multiplier - 1, 0)
        log(f"\nDataset Preparation Complete!")
        log(f"Train samples: {len(train_samples)}")
        log(f"Val samples: {len(val_samples)}")
        log(f"Total original: {len(all_samples)} | Augmented copies added: {aug_pos}")
        log(f"CSV manifest saved to: {csv_path.replace(os.sep, '/')}")
        log(f"YAML config saved to: {self.yaml_path.replace(os.sep, '/')}")
        step(100, "Complete!")

        # 7. Always export COCO format (needed by RF-DETR; uses hard links so no space waste)
        log("\nExporting COCO format dataset...")
        step(96, "Exporting COCO format...")
        self.export_coco(log)

        # 8. Also write detection-format labels (bbox derived from polygons) for RT-DETR
        if self.task == 'segment':
            self._export_detect_variant(log)
            step(98, "Detection labels exported.")

    def export_coco(self, log=print):
        # Converts the prepared YOLO split into COCO JSON format under data/coco/.
        # Bounding boxes are derived from the polygon label coords.
        # Required by RF-DETR which expects COCO-style annotations.
        coco_root = "data/coco"
        categories = [{"id": 1, "name": "fabella", "supercategory": "anatomy"}]

        # RF-DETR (Roboflow layout) uses 'valid' for validation, not 'val'
        split_map = {"train": "train", "val": "valid"}
        for split in ["train", "val"]:
            split_dst = split_map[split]
            img_src  = os.path.join(self.base_output, split, "images")
            lbl_src  = os.path.join(self.base_output, split, "labels")
            img_dst  = os.path.join(coco_root, split_dst)
            ann_dst  = os.path.join(coco_root, "annotations")

            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(ann_dst, exist_ok=True)

            if not os.path.exists(img_src):
                log(f"  Skipping COCO {split_dst}: {img_src} not found.")
                continue

            coco = {"images": [], "annotations": [], "categories": categories}
            img_id  = 0
            ann_id  = 0

            for fname in sorted(os.listdir(img_src)):
                if not fname.lower().endswith('.png'):
                    continue

                src_path = os.path.join(img_src, fname)
                dst_path = os.path.join(img_dst, fname)
                if not os.path.exists(dst_path):
                    try:
                        os.link(src_path, dst_path)   # hard link — zero extra disk space
                    except OSError:
                        shutil.copy(src_path, dst_path)  # fallback for cross-drive

                img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                h, w = img.shape[:2]
                img_id += 1
                coco["images"].append({"id": img_id, "file_name": fname,
                                        "width": w, "height": h})

                # Find matching label
                stem    = os.path.splitext(fname)[0]
                lbl_path = os.path.join(lbl_src, stem + ".txt")
                labels  = augmentor.read_labels(lbl_path) if os.path.exists(lbl_path) else []

                for cid, coords in labels:
                    pts = augmentor.pair_points(coords)
                    # De-normalise to pixel coords
                    px_pts = [(x * w, y * h) for x, y in pts]
                    xs = [p[0] for p in px_pts]
                    ys = [p[1] for p in px_pts]
                    x_min, y_min = max(0, min(xs)), max(0, min(ys))
                    bw = min(w, max(xs)) - x_min
                    bh = min(h, max(ys)) - y_min
                    area   = bw * bh
                    # Flat segmentation polygon for COCO
                    seg    = [v for p in px_pts for v in (round(p[0], 2), round(p[1], 2))]
                    ann_id += 1
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,  # fabella
                        "segmentation": [seg],
                        "bbox": [round(x_min, 2), round(y_min, 2), round(bw, 2), round(bh, 2)],
                        "area": round(area, 2),
                        "iscrowd": 0,
                    })

            # Standard path (for reference / other tools)
            json_path = os.path.join(ann_dst, f"instances_{split}.json")
            with open(json_path, "w") as f:
                json.dump(coco, f)

            # RF-DETR / Roboflow COCO layout: _annotations.coco.json inside each split folder
            rfdetr_json_path = os.path.join(img_dst, "_annotations.coco.json")
            with open(rfdetr_json_path, "w") as f:
                json.dump(coco, f)

            log(f"  COCO {split_dst}: {img_id} images, {ann_id} annotations -> {json_path.replace(os.sep, '/')}")

        # RF-DETR also requires a test/ split. Mirror valid/ -> test/.
        valid_dir = os.path.join(coco_root, "valid")
        test_dir  = os.path.join(coco_root, "test")
        if os.path.isdir(valid_dir):
            if os.path.isdir(test_dir):
                shutil.rmtree(test_dir)
            shutil.copytree(valid_dir, test_dir)
            log(f"  COCO test: mirrored from valid/ -> data/coco/test/")

    def _export_detect_variant(self, log=print):
        """Create detection (bbox) labels + YAML from existing seg labels.

        This allows RT-DETR and other detection models to train
        without re-running the full prepare step.  Images are hard-linked
        (zero extra disk space); only the label files are new.
        """
        seg_base = self.base_output  # e.g. data/yolo/seg
        det_base = seg_base.replace('/seg', '/det').replace('\\seg', '\\det')
        if det_base == seg_base:
            det_base = os.path.join('data', 'yolo', 'det')

        for split in ['train', 'val']:
            seg_img_dir = os.path.join(seg_base, split, 'images')
            seg_lbl_dir = os.path.join(seg_base, split, 'labels')
            det_img_dir = os.path.join(det_base, split, 'images')
            det_lbl_dir = os.path.join(det_base, split, 'labels')
            os.makedirs(det_img_dir, exist_ok=True)
            os.makedirs(det_lbl_dir, exist_ok=True)

            if not os.path.isdir(seg_img_dir):
                continue

            # Hard-link images (zero extra space)
            for fname in os.listdir(seg_img_dir):
                src = os.path.join(seg_img_dir, fname)
                dst = os.path.join(det_img_dir, fname)
                if not os.path.exists(dst):
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy(src, dst)

            # Convert polygon labels → bbox labels
            if not os.path.isdir(seg_lbl_dir):
                continue
            for fname in os.listdir(seg_lbl_dir):
                if not fname.endswith('.txt'):
                    continue
                seg_labels = augmentor.read_labels(os.path.join(seg_lbl_dir, fname))
                det_labels = [(cid, self._poly_to_bbox(coords)) for cid, coords in seg_labels]
                augmentor.write_labels(os.path.join(det_lbl_dir, fname), det_labels)

        # Write detection YAML
        det_yaml = os.path.join('data', 'yolo', 'data_det.yaml')
        yaml_content = f"""path: {os.path.abspath(det_base).replace(os.sep, '/')}
train: train/images
val: val/images

names:
  0: fabella
"""
        with open(det_yaml, 'w') as f:
            f.write(yaml_content)
        log(f"  Detection variant: {det_base} + {det_yaml}")
