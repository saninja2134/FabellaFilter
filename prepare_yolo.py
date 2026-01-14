import os
import shutil
import random
import pydicom
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def convert_dicom_to_16bit_png(src_path, dst_path):
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

def setup_dataset():
    pos_img_dir = "dataset_sorted/pos"
    label_dir = "labels/pos"
    neg_dicom_dir = "neg"
    
    # 1. Gather all POS samples that have labels
    labeled_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    pos_samples = []
    
    if not os.path.exists(pos_img_dir):
        print(f"Error: {pos_img_dir} does not exist.")
        return

    for f in labeled_files:
        png_name = f.replace('.txt', '.png')
        png_path = os.path.join(pos_img_dir, png_name)
        if os.path.exists(png_path):
            pos_samples.append((png_path, os.path.join(label_dir, f)))
    
    if not pos_samples:
        print("No labeled positive images found. Have you finished labeling?")
        return
        
    print(f"Found {len(pos_samples)} labeled positive images.")
    
    # 2. Pick a random set of NEG images
    neg_dicoms = [f for f in os.listdir(neg_dicom_dir) if f.lower().endswith('.dcm')]
    random.shuffle(neg_dicoms)
    # We'll use a 1:1 ratio for simplicity, or 1:2 if you prefer. 1:1 is good for a start.
    num_neg = min(len(neg_dicoms), len(pos_samples))
    selected_neg = neg_dicoms[:num_neg]
    
    print(f"Processing {num_neg} negative background images...")
    
    temp_neg_png = "temp_neg_png"
    if not os.path.exists(temp_neg_png): os.makedirs(temp_neg_png)
    
    neg_samples = []
    for dcm in selected_neg:
        png_name = dcm.replace('.dcm', '.png')
        dst_path = os.path.join(temp_neg_png, png_name)
        if convert_dicom_to_16bit_png(os.path.join(neg_dicom_dir, dcm), dst_path):
            neg_samples.append((dst_path, None))

    # 3. Split into Train/Val
    all_samples = pos_samples + neg_samples
    train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    
    # 4. Move to YOLO structure
    base = "yolo_dataset"
    if os.path.exists(base): shutil.rmtree(base) # Reset
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base, split, 'labels'), exist_ok=True)
        
    def copy_samples(samples, split):
        for img_path, lbl_path in samples:
            fname = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(base, split, 'images', fname))
            if lbl_path:
                shutil.copy(lbl_path, os.path.join(base, split, 'labels', os.path.basename(lbl_path)))
            else:
                # Ultralytics treats missing label files as background (negative) samples.
                pass
                
    print("Building training set...")
    copy_samples(train_samples, 'train')
    print("Building validation set...")
    copy_samples(val_samples, 'val')
    
    # 5. Create data.yaml
    yaml_content = f"""path: {os.path.abspath(base)}
train: train/images
val: val/images

names:
  0: fabella
"""
    with open("data.yaml", 'w') as f:
        f.write(yaml_content)
        
    print(f"\nDataset Preparation Complete!")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Total: {len(all_samples)}")
    print(f"Config saved to: data.yaml")

if __name__ == "__main__":
    setup_dataset()
