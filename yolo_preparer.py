"""
Module for preparing the dataset for YOLO training.
"""
import os
import shutil
import random
import pydicom
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class YoloPreparer:
    """
    A class to prepare and split the dataset for YOLO model training.
    """
    def __init__(self, pos_img_dir="dataset_sorted/pos", label_dir="labels/pos", neg_dicom_dir="neg", base_output="yolo_dataset"):
        """
        Initializes the YoloPreparer.
        
        Args:
            pos_img_dir (str): Directory containing positive images.
            label_dir (str): Directory containing labels.
            neg_dicom_dir (str): Directory containing negative DICOM images.
            base_output (str): Output directory for the YOLO dataset.
        """
        self.pos_img_dir = pos_img_dir
        self.label_dir = label_dir
        self.neg_dicom_dir = neg_dicom_dir
        self.base_output = base_output

    def convert_dicom_to_16bit_png(self, src_path, dst_path):
        """
        Converts a single DICOM file to a 16-bit PNG.
        
        Args:
            src_path (str): Path to the source DICOM file.
            dst_path (str): Path to save the converted PNG file.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            ds = pydicom.dcmread(src_path)
            img = ds.pixel_array.astype(float)
            
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

    def setup_dataset(self, progress_callback=None):
        """
        Prepares the dataset by gathering positive samples, sampling negative samples,
        splitting into train/val sets, and creating the data.yaml file.
        
        Args:
            progress_callback (callable, optional): A function to call with progress updates.
        """
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

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
            if os.path.exists(png_path):
                pos_samples.append((png_path, os.path.join(self.label_dir, f)))
        
        if not pos_samples:
            log("No labeled positive images found. Have you finished labeling?")
            return
            
        log(f"Found {len(pos_samples)} labeled positive images.")
        
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
        
        neg_samples = []
        for dcm in selected_neg:
            png_name = dcm.replace('.dcm', '.png')
            dst_path = os.path.join(temp_neg_png, png_name)
            if self.convert_dicom_to_16bit_png(os.path.join(self.neg_dicom_dir, dcm), dst_path):
                neg_samples.append((dst_path, None))

        # 3. Split into Train/Val
        all_samples = pos_samples + neg_samples
        train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
        
        # 4. Move to YOLO structure
        if os.path.exists(self.base_output): shutil.rmtree(self.base_output) # Reset
        
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.base_output, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.base_output, split, 'labels'), exist_ok=True)
            
        def copy_samples(samples, split):
            for img_path, lbl_path in samples:
                fname = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(self.base_output, split, 'images', fname))
                if lbl_path:
                    shutil.copy(lbl_path, os.path.join(self.base_output, split, 'labels', os.path.basename(lbl_path)))
                    
        log("Building training set...")
        copy_samples(train_samples, 'train')
        log("Building validation set...")
        copy_samples(val_samples, 'val')
        
        # 5. Create data.yaml
        yaml_content = f"""path: {os.path.abspath(self.base_output)}
train: train/images
val: val/images

names:
  0: fabella
"""
        with open("data.yaml", 'w') as f:
            f.write(yaml_content)
            
        log(f"\nDataset Preparation Complete!")
        log(f"Train samples: {len(train_samples)}")
        log(f"Val samples: {len(val_samples)}")
        log(f"Total: {len(all_samples)}")
        log(f"Config saved to: data.yaml")
