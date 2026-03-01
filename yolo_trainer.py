# Module for training the YOLO model.
from ultralytics import YOLO
import torch
import os

class YoloTrainer:
    # A class to handle the training of the YOLO model.
    def __init__(self, task="segment", model_version="11", epochs=100, imgsz=1024, batch=4):
        # Initializes the YoloTrainer.
        # Args:
        # task (str): The YOLO task (e.g., 'segment', 'obb').
        # model_version (str): The YOLO version (e.g., '8', '11').
        # epochs (int): Number of training epochs.
        # imgsz (int): Image size for training.
        # batch (int): Batch size for training.
        self.task = task
        self.data_yaml = f"data/yolo/data_{'obb' if task == 'obb' else 'seg'}.yaml"
        
        task_suffix = "-obb" if task == "obb" else "-seg"
        
        # Handle YOLOv8 naming convention (yolov8) vs others (yolo11, yolo26)
        prefix = "yolov" if model_version == "8" else "yolo"
        
        self.model_name = f"{prefix}{model_version}n{task_suffix}.pt"
            
        self.fallback_model = f"yolov8n{task_suffix}.pt"
        
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch

    def train_fabella(self, progress_callback=None):
        # Starts the training process for the YOLO model.
        # Args:
        # progress_callback (callable, optional): A function to call with progress updates.
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

        if not os.path.exists(self.data_yaml):
            log(f"Error: {self.data_yaml} not found. Please run 'Prepare YOLO' first.")
            return

        # 1. Hardware check
        device = 0 if torch.cuda.is_available() else 'cpu'
        log(f"--- Hardware Status ---")
        log(f"Device: {device}")
        if device == 0:
            log(f"GPU: {torch.cuda.get_device_name(0)}")
            log(f"CUDA Version: {torch.version.cuda}")
        log("-----------------------\n")

        # 2. Load Model
        log(f"Initializing {self.model_name}...")
        try:
            model = YOLO(self.model_name)
        except Exception as e:
            log(f"Error loading {self.model_name}: {e}")
            log(f"Warning: Could not load {self.model_name}. Standard weights might not be available yet.")
            log(f"Falling back to SOTA {self.fallback_model} as fallback.")
            model = YOLO(self.fallback_model)

        # 3. Train
        log(f"Starting {self.task} training...")
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=device,
            workers=4,
            project="output/runs",
            name=f"fabella_{self.task}_v1",
            patience=20,
            save=True,
            fliplr=0.5,
            flipud=0.2,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            mixup=0.1,
            mosaic=1.0
        )

        log("\nTraining Finished!")
        log(f"Best model saved in: output/runs/fabella_{self.task}_v1/weights/best.pt")
