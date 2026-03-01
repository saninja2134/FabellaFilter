# Module for testing the trained YOLO model.
import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil

class YoloTester:
    # A class to handle testing the trained YOLO model on a dataset.
    def __init__(self, task="segment", src_dir="data/png/pos", sorted_dir="data/sorted/pos"):
        # Initializes the YoloTester.
        # Args:
        # task (str): The YOLO task ('segment' or 'obb').
        # src_dir (str): Directory containing source images for testing.
        # sorted_dir (str): Directory containing sorted images to exclude from testing.
        self.task = task
        self.model_path = f"output/runs/fabella_{'obb' if task == 'obb' else 'segment'}_v1/weights/best.pt"
        self.src_dir = src_dir
        self.sorted_dir = sorted_dir
        self.output_dir = f"output/test_{'obb' if task == 'obb' else 'seg'}"

    def run_test(self, progress_callback=None):
        # Runs the test process using the trained YOLO model.
        # Args:
        # progress_callback (callable, optional): A function to call with progress updates.
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

        # 1. Config
        if not os.path.exists(self.model_path):
            log(f"Error: {self.model_path} not found.")
            return

        # Subfolders
        det_dir = os.path.join(self.output_dir, "detected")
        undet_dir = os.path.join(self.output_dir, "undetected")

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(det_dir)
        os.makedirs(undet_dir)

        # 2. Filter images
        if not os.path.exists(self.src_dir):
            log(f"Error: {self.src_dir} not found.")
            return

        all_images = sorted([f for f in os.listdir(self.src_dir) if f.lower().endswith('.png')])
        sorted_images = set(os.listdir(self.sorted_dir)) if os.path.exists(self.sorted_dir) else set()
        unsorted = [f for f in all_images if f not in sorted_images]
        
        if not unsorted:
            log("No unsorted images found.")
            return

        midpoint = len(unsorted) // 2
        test_batch = unsorted[midpoint:]
        
        # 3. Load Model
        log(f"Loading model from {self.model_path}...")
        model = YOLO(self.model_path)
        log(f"Processing {len(test_batch)} images...")

        for i, img_name in enumerate(test_batch):
            if i >= 1000: break
            
            img_path = os.path.join(self.src_dir, img_name)
            results = model.predict(source=img_path, conf=0.25, imgsz=1024, verbose=False)
            result = results[0]

            # Load image for custom visualization
            raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if raw_img is None: continue
            
            if raw_img.dtype == np.uint16:
                img_8 = (raw_img / 256).astype(np.uint8)
            else:
                img_8 = raw_img
            
            img_8 = cv2.normalize(img_8, None, 0, 255, cv2.NORM_MINMAX)
            if len(img_8.shape) == 2:
                img_visual = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
            else:
                img_visual = img_8.copy()

            has_detection = False
            if self.task == 'obb':
                if hasattr(result, 'obb') and result.obb is not None and len(result.obb.xyxyxyxy) > 0:
                    has_detection = True
                    for box, conf in zip(result.obb.xyxyxyxy, result.obb.conf):
                        pts = box.cpu().numpy().astype(np.int32)
                        cv2.polylines(img_visual, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        right_most_x = np.max(pts[:, 0])
                        avg_y = int(np.mean(pts[:, 1]))
                        
                        label = f"{conf:.2f}"
                        cv2.putText(img_visual, label, (right_most_x + 15, avg_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                if hasattr(result, 'masks') and result.masks is not None and len(result.masks.xy) > 0:
                    has_detection = True
                    overlay = img_visual.copy()
                    for mask, conf in zip(result.masks.xy, result.boxes.conf):
                        pts = mask.astype(np.int32)
                        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                        cv2.polylines(img_visual, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
                        
                        if len(pts) > 0:
                            right_most_x = np.max(pts[:, 0])
                            avg_y = int(np.mean(pts[:, 1]))
                            
                            label = f"{conf:.2f}"
                            cv2.putText(img_visual, label, (right_most_x + 15, avg_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    cv2.addWeighted(overlay, 0.3, img_visual, 0.7, 0, img_visual)

            save_name = os.path.join(det_dir if has_detection else undet_dir, img_name)
            cv2.imwrite(save_name, img_visual)

            if (i+1) % 50 == 0:
                log(f"Processed {i+1}/{len(test_batch)}...")

        log(f"\nTest complete!")
        log(f"Detected: {len(os.listdir(det_dir))}")
        log(f"Undetected: {len(os.listdir(undet_dir))}")
        log(f"Results are in {self.output_dir}/")
