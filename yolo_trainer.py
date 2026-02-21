"""
Module for training the YOLO model.
"""
from ultralytics import YOLO
import torch
import os

class YoloTrainer:
    """
    A class to handle the training of the YOLO model.
    """
    def __init__(self, data_yaml="data.yaml", model_name="yolo12n-obb.pt", fallback_model="yolo11n-obb.pt", epochs=100, imgsz=1024, batch=4):
        """
        Initializes the YoloTrainer.
        
        Args:
            data_yaml (str): Path to the data configuration file.
            model_name (str): Name of the primary YOLO model to use.
            fallback_model (str): Name of the fallback YOLO model.
            epochs (int): Number of training epochs.
            imgsz (int): Image size for training.
            batch (int): Batch size for training.
        """
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch

    def train_fabella(self, progress_callback=None):
        """
        Starts the training process for the YOLO model.
        
        Args:
            progress_callback (callable, optional): A function to call with progress updates.
        """
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
            log(f"Warning: Could not load {self.model_name}. Standard weights might not be available yet.")
            log(f"Falling back to SOTA {self.fallback_model} as fallback.")
            model = YOLO(self.fallback_model)

        # 3. Train
        log("Starting training...")
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=device,
            workers=4,
            name="fabella_obb_v12",
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
        log(f"Best model saved in: runs/obb/fabella_obb_v12/weights/best.pt")
