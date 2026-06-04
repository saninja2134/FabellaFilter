# Module for converting DICOM files to PNG format.
import pydicom
import numpy as np
import cv2
import os
import sys
import csv

class DicomConverter:
    # A class to handle the conversion of DICOM images to 16-bit PNG format.
    def __init__(self, base_dir="data/raw", output_base="data/png"):
        self.base_dir = base_dir
        self.output_base = output_base

    def check_dependencies(self):
        # Checks for required DICOM decompression handlers.
        handlers = []
        try:
            from pydicom.pixel_data_handlers import gdcm_handler, pylibjpeg_handler
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

    @staticmethod
    def _dicom_to_uint16(ds):
        """Convert a pydicom Dataset pixel array to windowed 16-bit PNG array."""
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

        return img

    def convert_to_bone_png(self, dicom_path, output_folder, progress_callback=None):
        # Converts DICOM files in a directory to 16-bit PNG.
        self.check_dependencies()
        os.makedirs(output_folder, exist_ok=True)

        dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
        total_files = len(dicom_files)
        print(f"Found {total_files} DICOM files. Starting conversion...")

        demographics = []

        for i, filename in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(os.path.join(dicom_path, filename))
                img = self._dicom_to_uint16(ds)

                png_name = os.path.splitext(filename)[0] + ".png"
                cv2.imwrite(os.path.join(output_folder, png_name), img)

                # Extract demographic data
                _g = lambda tag: str(getattr(ds, tag, 'UNKNOWN'))
                demographics.append({
                    'FileName': png_name,
                    'PatientID': _g('PatientID'),
                    'PatientName': _g('PatientName'),
                    'PatientAge': _g('PatientAge'),
                    'PatientBirthDate': _g('PatientBirthDate'),
                    'PatientSex': _g('PatientSex'),
                    'PatientAddress': _g('PatientAddress'),
                    'EthnicGroup': _g('EthnicGroup'),
                    'StudyDate': _g('StudyDate'),
                    'StudyDescription': _g('StudyDescription'),
                    'Modality': _g('Modality'),
                    'BodyPartExamined': _g('BodyPartExamined'),
                    'Laterality': _g('Laterality'),
                    'ViewPosition': _g('ViewPosition'),
                    'Manufacturer': _g('Manufacturer'),
                    'ManufacturerModelName': _g('ManufacturerModelName'),
                })

                if progress_callback:
                    progress_callback(f"Converted {i+1}/{total_files}: {filename}")

            except Exception as e:
                msg = f"Error converting {filename}: {e}"
                print(msg)
                if progress_callback:
                    progress_callback(msg)

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
