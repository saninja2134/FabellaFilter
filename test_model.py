import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil

def run_test():
    # 1. Config
    model_path = "runs/obb/fabella_obb_v12/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    src_dir = "dataset_png/pos"
    sorted_dir = "dataset_sorted/pos"
    output_dir = "test_output"
    
    # Subfolders
    det_dir = os.path.join(output_dir, "detected")
    undet_dir = os.path.join(output_dir, "undetected")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(det_dir)
    os.makedirs(undet_dir)

    # 2. Filter images
    all_images = sorted([f for f in os.listdir(src_dir) if f.lower().endswith('.png')])
    sorted_images = set(os.listdir(sorted_dir)) if os.path.exists(sorted_dir) else set()
    unsorted = [f for f in all_images if f not in sorted_images]
    
    if not unsorted:
        print("No unsorted images found.")
        return

    midpoint = len(unsorted) // 2
    test_batch = unsorted[midpoint:]
    
    # 3. Load Model
    model = YOLO(model_path)
    print(f"Processing {len(test_batch)} images...")

    for i, img_name in enumerate(test_batch):
        if i >= 1000: break
        
        img_path = os.path.join(src_dir, img_name)
        results = model.predict(source=img_path, conf=0.25, imgsz=1024, verbose=False)
        result = results[0]

        # Load image for custom visualization
        # Handling 16-bit for display
        raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None: continue
        
        if raw_img.dtype == np.uint16:
            # Normalize to 8-bit for viewing
            img_8 = (raw_img / 256).astype(np.uint8)
        else:
            img_8 = raw_img
        
        # Auto-contrast boost
        img_8 = cv2.normalize(img_8, None, 0, 255, cv2.NORM_MINMAX)
        if len(img_8.shape) == 2:
            img_visual = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
        else:
            img_visual = img_8.copy()

        has_detection = False
        if hasattr(result, 'obb') and result.obb is not None and len(result.obb.xyxyxyxy) > 0:
            has_detection = True
            for box, conf in zip(result.obb.xyxyxyxy, result.obb.conf):
                pts = box.cpu().numpy().astype(np.int32)
                # Draw Box
                cv2.polylines(img_visual, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Probability text position (to the right of the right-most point)
                right_most_x = np.max(pts[:, 0])
                avg_y = int(np.mean(pts[:, 1]))
                
                label = f"{conf:.2f}"
                # Move 15 pixels right to avoid overlapping the object
                cv2.putText(img_visual, label, (right_most_x + 15, avg_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Save to appropriate folder
        save_name = os.path.join(det_dir if has_detection else undet_dir, img_name)
        cv2.imwrite(save_name, img_visual)

        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_batch)}...")

    print(f"\nTest complete!")
    print(f"Detected: {len(os.listdir(det_dir))}")
    print(f"Undetected: {len(os.listdir(undet_dir))}")
    print(f"Results are in {output_dir}/")

if __name__ == "__main__":
    run_test()
