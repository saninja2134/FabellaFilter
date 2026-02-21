"""
Module for converting DICOM files to PNG format.
"""
import pydicom
import numpy as np
import cv2
import os
import sys

class DicomConverter:
    """
    A class to handle the conversion of DICOM images to 16-bit PNG format.
    """
    def __init__(self, base_dir=".", output_base="dataset_png"):
        """
        Initializes the DicomConverter.
        
        Args:
            base_dir (str): The base directory containing 'neg' and 'pos' folders.
            output_base (str): The base directory to save the converted PNGs.
        """
        self.base_dir = base_dir
        self.output_base = output_base

    def check_dependencies(self):
        """
        Checks for required DICOM decompression handlers.
        """
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
        """
        Converts DICOM files in a directory to PNG.
        
        Args:
            dicom_path (str): Path to the directory containing DICOM files.
            output_folder (str): Path to the directory to save PNG files.
            progress_callback (callable, optional): A function to call with progress updates.
        """
        self.check_dependencies()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
        total_files = len(dicom_files)
        
        print(f"Found {total_files} DICOM files. Starting conversion...")

        for i, filename in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(os.path.join(dicom_path, filename))
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

                png_name = os.path.splitext(filename)[0] + ".png"
                cv2.imwrite(os.path.join(output_folder, png_name), img)
                
                if progress_callback:
                    progress_callback(f"Converted {i+1}/{total_files}: {filename}")
                
            except Exception as e:
                print(f"Error converting {filename}: {e}")
                if progress_callback:
                    progress_callback(f"Error converting {filename}: {e}")

        print("Conversion complete!")

    def run_conversion(self, progress_callback=None):
        """
        Runs the conversion for both 'neg' and 'pos' folders.
        
        Args:
            progress_callback (callable, optional): A function to call with progress updates.
        """
        for category in ["neg", "pos"]:
            input_folder = os.path.join(self.base_dir, category)
            output_folder = os.path.join(self.output_base, category)
            
            if os.path.exists(input_folder):
                msg = f"\nProcessing {category} folder..."
                print(msg)
                if progress_callback: progress_callback(msg)
                self.convert_to_bone_png(input_folder, output_folder, progress_callback)
            else:
                msg = f"\nSkipping {category} - folder not found."
                print(msg)
                if progress_callback: progress_callback(msg)
