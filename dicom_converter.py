# Module for converting DICOM files to PNG format.
import pydicom
import numpy as np
import cv2
import os
import sys
import torch
import csv

class DicomConverter:
    # A class to handle the conversion of DICOM images to 16-bit PNG format.
    def __init__(self, base_dir=".", output_base="dataset_png"):
        # Initializes the DicomConverter.
        # 
        # Args:
        #     base_dir (str): The base directory containing 'neg' and 'pos' folders.
        #     output_base (str): The base directory to save the converted PNGs.
        self.base_dir = base_dir
        self.output_base = output_base

    def check_dependencies(self):
        # Checks for required DICOM decompression handlers.
        print(f"Running with Python: {sys.executable}")
        handlers = []
        try:
            from pydicom.pixel_data_handlers import gdcm_handler, pylibjpeg_handler, pillow_handler
            if gdcm_handler.is_available(): handlers.append("gdcm")
            if pylibjpeg_handler.is_available(): handlers.append("pylibjpeg")
            
            try:
                import imagecodecs
                handlers.append("imagecodecs")
            except ImportError:
                pass
                
        except ImportError:
            pass
        
        if not handlers:
            print("WARNING: No DICOM decompression handlers detected!")
        else:
            print(f"Detected handlers: {', '.join(handlers)}")

    def convert_to_bone_png(self, dicom_path, output_folder, progress_callback=None):
        # Converts DICOM files in a directory to PNG.
        # 
        # Args:
        #     dicom_path (str): Path to the directory containing DICOM files.
        #     output_folder (str): Path to the directory to save PNG files.
        #     progress_callback (callable, optional): A function to call with progress updates.
        self.check_dependencies()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
        total_files = len(dicom_files)
        
        print(f"Found {total_files} DICOM files. Starting conversion...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("Using GPU for image processing.")
        else:
            print("Using CPU for image processing.")

        demographics = []

        for i, filename in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(os.path.join(dicom_path, filename))
                img = ds.pixel_array.astype(float)
                
                # Move to GPU if available
                if device.type == "cuda":
                    img_tensor = torch.from_numpy(img).to(device)
                else:
                    img_tensor = img

                if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
                    if device.type == "cuda":
                        img_tensor = img_tensor * ds.RescaleSlope + ds.RescaleIntercept
                    else:
                        img_tensor = img_tensor * ds.RescaleSlope + ds.RescaleIntercept

                if 'WindowCenter' in ds and 'WindowWidth' in ds:
                    wc = ds.WindowCenter
                    ww = ds.WindowWidth
                    if hasattr(wc, '__iter__'): wc = wc[0]
                    if hasattr(ww, '__iter__'): ww = ww[0]
                    img_min = float(wc) - float(ww) // 2
                    img_max = float(wc) + float(ww) // 2
                else:
                    if device.type == "cuda":
                        # PyTorch doesn't have a direct percentile function for 2D tensors that works exactly like numpy's
                        # We can approximate or move back to CPU for this step if needed, but for speed, let's use quantile
                        img_flat = img_tensor.flatten()
                        img_min = torch.quantile(img_flat, 0.01).item()
                        img_max = torch.quantile(img_flat, 0.99).item()
                    else:
                        img_min = np.percentile(img_tensor, 1)
                        img_max = np.percentile(img_tensor, 99)

                if device.type == "cuda":
                    img_tensor = torch.clamp(img_tensor, img_min, img_max)
                    img_tensor = ((img_tensor - img_min) / (img_max - img_min) * 65535.0).to(torch.int32) # Use int32 before converting to numpy uint16
                    img = img_tensor.cpu().numpy().astype(np.uint16)
                else:
                    img_tensor = np.clip(img_tensor, img_min, img_max)
                    img = ((img_tensor - img_min) / (img_max - img_min) * 65535.0).astype(np.uint16)

                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = 65535 - img

                png_name = os.path.splitext(filename)[0] + ".png"
                cv2.imwrite(os.path.join(output_folder, png_name), img)
                
                # Extract demographic data
                demo_data = {
                    'FileName': png_name,
                    'PatientID': str(getattr(ds, 'PatientID', 'UNKNOWN')),
                    'PatientName': str(getattr(ds, 'PatientName', 'UNKNOWN')),
                    'PatientAge': str(getattr(ds, 'PatientAge', 'UNKNOWN')),
                    'PatientBirthDate': str(getattr(ds, 'PatientBirthDate', 'UNKNOWN')),
                    'PatientSex': str(getattr(ds, 'PatientSex', 'UNKNOWN')),
                    'PatientAddress': str(getattr(ds, 'PatientAddress', 'UNKNOWN')),
                    'EthnicGroup': str(getattr(ds, 'EthnicGroup', 'UNKNOWN')),
                    'OtherPatientIDs': str(getattr(ds, 'OtherPatientIDs', 'UNKNOWN')),
                    'IssuerOfPatientID': str(getattr(ds, 'IssuerOfPatientID', 'UNKNOWN')),
                    'CurrentPatientLocation': str(getattr(ds, 'CurrentPatientLocation', 'UNKNOWN')),
                    'StudyDate': str(getattr(ds, 'StudyDate', 'UNKNOWN')),
                    'StudyTime': str(getattr(ds, 'StudyTime', 'UNKNOWN')),
                    'StudyDescription': str(getattr(ds, 'StudyDescription', 'UNKNOWN')),
                    'Modality': str(getattr(ds, 'Modality', 'UNKNOWN')),
                    'BodyPartExamined': str(getattr(ds, 'BodyPartExamined', 'UNKNOWN')),
                    'Laterality': str(getattr(ds, 'Laterality', 'UNKNOWN')),
                    'ViewPosition': str(getattr(ds, 'ViewPosition', 'UNKNOWN')),
                    'ReferringPhysicianName': str(getattr(ds, 'ReferringPhysicianName', 'UNKNOWN')),
                    'RequestingPhysician': str(getattr(ds, 'RequestingPhysician', 'UNKNOWN')),
                    'Manufacturer': str(getattr(ds, 'Manufacturer', 'UNKNOWN')),
                    'ManufacturerModelName': str(getattr(ds, 'ManufacturerModelName', 'UNKNOWN')),
                    'StationName': str(getattr(ds, 'StationName', 'UNKNOWN'))
                }
                demographics.append(demo_data)
                
                if progress_callback:
                    progress_callback(f"Converted {i+1}/{total_files}: {filename}")
                
            except Exception as e:
                print(f"Error converting {filename}: {e}")
                if progress_callback:
                    progress_callback(f"Error converting {filename}: {e}")

        print("Conversion complete!")
        return demographics

    def run_conversion(self, progress_callback=None):
        # Runs the conversion for both 'neg' and 'pos' folders.
        # 
        # Args:
        #     progress_callback (callable, optional): A function to call with progress updates.
        all_demographics = []
        for category in ["neg", "pos"]:
            input_folder = os.path.join(self.base_dir, category)
            output_folder = os.path.join(self.output_base, category)
            
            if os.path.exists(input_folder):
                msg = f"\nProcessing {category} folder..."
                print(msg)
                if progress_callback: progress_callback(msg)
                category_demographics = self.convert_to_bone_png(input_folder, output_folder, progress_callback)
                if category_demographics:
                    all_demographics.extend(category_demographics)
            else:
                msg = f"\nSkipping {category} - folder not found."
                print(msg)
                if progress_callback: progress_callback(msg)

        if all_demographics:
            csv_path = os.path.join(self.output_base, "demographics.csv")
            try:
                keys = all_demographics[0].keys()
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(all_demographics)
                msg = f"\nDemographics saved to {csv_path}"
                print(msg)
                if progress_callback: progress_callback(msg)
            except Exception as e:
                msg = f"\nError saving demographics to CSV: {e}"
                print(msg)
                if progress_callback: progress_callback(msg)
