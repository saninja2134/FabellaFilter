# Module for preparing the dataset for YOLO training.
import os
import shutil
import random
import csv
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
        # task (str): The task type ('segment' or 'obb').
        # pos_img_dir (str): Directory containing positive images.
        # neg_dicom_dir (str): Directory containing negative DICOM images.
        self.task = task
        self.pos_img_dir = pos_img_dir
        self.label_dir = f"data/labels/{'obb' if task == 'obb' else 'seg'}"
        self.neg_dicom_dir = neg_dicom_dir
        self.base_output = f"data/yolo/{'obb' if task == 'obb' else 'seg'}"
        self.yaml_path = f"data/yolo/data_{'obb' if task == 'obb' else 'seg'}.yaml"

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

    def setup_dataset(self, config=None, progress_callback=None):
        # Prepares the dataset by gathering positive samples, sampling negative samples,
        # splitting into train/val sets, and creating the data.yaml file.
        #
        # Args:
        # config (dict, optional): Augmentation/preprocessing config from DatasetGeneratorModal.
        # progress_callback (callable, optional): A function to call with progress updates.
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

        multiplier = 1
        if config:
            multiplier = config.get('generation', {}).get('multiplier', 1)

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

        # 2. Pick a random set of NEG images
        if not os.path.exists(self.neg_dicom_dir):
            log(f"Error: {self.neg_dicom_dir} does not exist.")
            return

        neg_dicoms = [f for f in os.listdir(self.neg_dicom_dir) if f.lower().endswith('.dcm')]
        random.shuffle(neg_dicoms)
        num_neg = min(len(neg_dicoms), len(pos_samples))
        selected_neg = neg_dicoms[:num_neg]

        log(f"Processing {num_neg} negative background images...")

        temp_neg_png = "temp_neg_png"
        if not os.path.exists(temp_neg_png): os.makedirs(temp_neg_png)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        neg_samples = []
        for i, dcm in enumerate(selected_neg):
            png_name = dcm.replace('.dcm', '.png')
            dst_path = os.path.join(temp_neg_png, png_name)
            if self.convert_dicom_to_16bit_png(os.path.join(self.neg_dicom_dir, dcm), dst_path, device=device):
                neg_samples.append((dst_path, None))

            if (i + 1) % 10 == 0 or (i + 1) == num_neg:
                log(f"Converted {i + 1}/{num_neg} negative images...")

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
        process_samples(train_samples, 'train')
        log("Building validation set...")
        process_samples(val_samples, 'val')

        # CLEANUP: Remove temporary negative DICOM conversions to prevent memory leaks
        if os.path.exists(temp_neg_png):
            shutil.rmtree(temp_neg_png)
            log("Cleaned up temporary negative PNGs.")

        # 5. Create data.yaml
        yaml_content = f"""path: {os.path.abspath(self.base_output)}
train: train/images
val: val/images

names:
  0: fabella
"""
        with open(self.yaml_path, 'w') as f:
            f.write(yaml_content)

        # 6. Write CSV manifest
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
        log(f"CSV manifest saved to: {csv_path}")
        log(f"YAML config saved to: {self.yaml_path}")
