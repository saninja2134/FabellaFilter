import os
import shutil
import random
import yaml
from pathlib import Path

def setup_yolo_dataset(pos_img_dir, neg_img_dir, label_dir, output_dir, val_split=0.2):
    # 1. Setup Folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 2. Identify Labeled Positives
    all_pos_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    labeled_stems = [os.path.splitext(f)[0] for f in all_pos_labels]
    
    pos_data = [] # List of (img_path, label_path)
    for stem in labeled_stems:
        img_path = os.path.join(pos_img_dir, stem + ".png")
        label_path = os.path.join(label_dir, stem + ".txt")
        if os.path.exists(img_path):
            pos_data.append((img_path, label_path))

    print(f"Found {len(pos_data)} labeled positive images.")

    # 3. Sample matching Negatives (Background images)
    all_neg_images = [f for f in os.listdir(neg_img_dir) if f.lower().endswith('.png')]
    num_to_sample = min(len(pos_data), len(all_neg_images))
    sampled_negs = random.sample(all_neg_images, num_to_sample)
    
    neg_data = [] # List of (img_path, None)
    for img_name in sampled_negs:
        neg_data.append((os.path.join(neg_img_dir, img_name), None))

    print(f"Sampled {len(neg_data)} negative background images for balance.")

    # 4. Combine and Shuffle
    all_data = pos_data + neg_data
    random.shuffle(all_data)

    # 5. Split and Copy
    split_idx = int(len(all_data) * (1 - val_split))
    splits = {
        'train': all_data[:split_idx],
        'val': all_data[split_idx:]
    }

    for split_name, data_list in splits.items():
        print(f"Copying {len(data_list)} files to {split_name}...")
        for img_path, label_path in data_list:
            filename = os.path.basename(img_path)
            # Copy Image
            shutil.copy2(img_path, os.path.join(output_dir, split_name, 'images', filename))
            
            # Copy Label (if exists)
            if label_path:
                shutil.copy2(label_path, os.path.join(output_dir, split_name, 'labels', os.path.basename(label_path)))
            else:
                # YOLO treats empty label files as background, or just missing files as background.
                # To be explicit, we can create an empty file or just leave it missing.
                # Standard practice is an empty .txt with the same name.
                label_filename = os.path.splitext(filename)[0] + ".txt"
                open(os.path.join(output_dir, split_name, 'labels', label_filename), 'w').close()

    # 6. Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'fabella'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("\nDataset preparation complete!")
    print(f"Data directory: {output_dir}")
    print("Created 'dataset.yaml' for YOLO training.")

if __name__ == "__main__":
    # Configure paths
    POS_IMG = "dataset_sorted/pos"
    NEG_IMG = "dataset_png/neg"
    LABELS = "labels/pos"
    OUTPUT = "yolo_ready_dataset"
    
    setup_yolo_dataset(POS_IMG, NEG_IMG, LABELS, OUTPUT)
