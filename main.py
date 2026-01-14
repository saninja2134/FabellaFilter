import pydicom
import numpy as np
import cv2
import os

import sys

# Ensure decompression handlers are registered
try:
    import pylibjpeg
except ImportError:
    pass
try:
    import gdcm
except ImportError:
    pass
try:
    import imagecodecs
except ImportError:
    pass

# Check for decompression support
def check_dependencies():
    print(f"Running with Python: {sys.executable}")
    handlers = []
    try:
        from pydicom.pixel_data_handlers import gdcm_handler, pylibjpeg_handler, pillow_handler
        if gdcm_handler.is_available(): handlers.append("gdcm")
        if pylibjpeg_handler.is_available(): handlers.append("pylibjpeg")
        
        # Check imagecodecs (often used by pydicom internally if available)
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

def convert_to_bone_png(dicom_path, output_folder):
    check_dependencies()
    # 1. Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Get list of DICOM files
    dicom_files = [f for f in os.listdir(dicom_path) if f.lower().endswith('.dcm')]
    
    print(f"Found {len(dicom_files)} DICOM files. Starting conversion...")

    for filename in dicom_files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_path, filename))
            img = ds.pixel_array.astype(float)

            # Apply Rescale Slope and Intercept (converts to Hounsfield Units)
            if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
                img = img * ds.RescaleSlope + ds.RescaleIntercept

            # --- SMART WINDOWING ---
            # Try to use DICOM's intended windowing first
            if 'WindowCenter' in ds and 'WindowWidth' in ds:
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                # Handle cases where multiple values are provided
                if hasattr(wc, '__iter__'): wc = wc[0]
                if hasattr(ww, '__iter__'): ww = ww[0]
                img_min = float(wc) - float(ww) // 2
                img_max = float(wc) + float(ww) // 2
            else:
                # Fallback: Auto-windowing using percentiles (99th percentile prevents white-out)
                img_min = np.percentile(img, 1)
                img_max = np.percentile(img, 99)

            # Clip and Normalize to 16-bit
            img = np.clip(img, img_min, img_max)
            img = ((img - img_min) / (img_max - img_min) * 65535.0).astype(np.uint16)

            # --- PHOTOMETRIC CORRECTION ---
            # MONOCHROME1: Bones are Black. MONOCHROME2: Bones are White.
            # We ensure "Bones are White" for YOLO labeling.
            if ds.PhotometricInterpretation == "MONOCHROME1":
                img = 65535 - img

            # 3. Save as PNG
            png_name = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(output_folder, png_name), img)
            
        except Exception as e:
            print(f"Error converting {filename}: {e}")

    print("Conversion complete! You can now upload these PNGs to your labeling tool.")

# EXECUTION: Process both 'neg' and 'pos' folders
if __name__ == "__main__":
    base_dir = "."  # Current directory where neg and pos folders are located
    output_base = "dataset_png"
    
    for category in ["neg", "pos"]:
        input_folder = os.path.join(base_dir, category)
        output_folder = os.path.join(output_base, category)
        
        if os.path.exists(input_folder):
            print(f"\nProcessing {category} folder...")
            convert_to_bone_png(input_folder, output_folder)
        else:
            print(f"\nSkipping {category} - folder not found.")