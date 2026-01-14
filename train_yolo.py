from ultralytics import YOLO
import torch
import os

def train_fabella():
    # 1. Hardware check
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"--- Hardware Status ---")
    print(f"Device: {device}")
    if device == 0:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("-----------------------\n")

    # 2. Load Model
    # The user specifically requested YOLOv12-OBB. 
    # If the architecture is recent, Ultralytics usually maintains 
    # the same API: YOLO('name.pt')
    model_name = "yolo12n-obb.pt"
    
    print(f"Initializing {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Warning: Could not load {model_name}. Standard weights might not be available yet.")
        print("Falling back to SOTA YOLOv11-OBB as fallback.")
        model = YOLO("yolo11n-obb.pt")

    # 3. Train
    # Medical images (PNG 16-bit) are high resolution. 
    # imgsz=1024 preserves bone texture better than 640.
    results = model.train(
        data="data.yaml",           # Path to the data config we created
        epochs=100,                 # 100 epochs is a good start
        imgsz=1024,                 # High resolution for fabella detection
        batch=4,                    # Adjust based on VRAM (4-8 is typical for 1024px)
        device=device,              # CUDA 0
        workers=4,                  # Multiprocessing for data loading
        name="fabella_obb_v12",     # Experiment name
        patience=20,                # Early stopping if no improvement
        save=True,                  # Save checkpoints
        # Augmentation settings tailored for X-rays (Medical)
        fliplr=0.5,                 # X-rays can be mirrored (left/right)
        flipud=0.2,                 # Top/down flip (less common but possible)
        hsv_h=0.0,                  # Disable color jitter (grayscale data)
        hsv_s=0.0,
        hsv_v=0.0,
        mixup=0.1,                  # Helps with small object detection
        mosaic=1.0                  # Combines multiple images
    )

    print("\nTraining Finished!")
    print(f"Best model saved in: runs/obb/fabella_obb_v12/weights/best.pt")

if __name__ == "__main__":
    # Ensure data.yaml exists before starting
    if not os.path.exists("data.yaml"):
        print("Error: data.yaml not found. Please run 'prepare_yolo.py' first.")
    else:
        train_fabella()
