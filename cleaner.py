import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

class FabellaCleaner:
    def __init__(self, root):
        self.root = root
        self.root.title("Fabella Dataset Cleaner")
        self.root.geometry("1000x900")

        # CONFIGURATION
        self.input_base = "dataset_png"
        self.output_base = "dataset_sorted"
        self.discard_base = "dataset_discarded"
        
        # Current category (running cleaning on 'pos' by default as requested)
        self.category = "pos" 
        self.input_dir = os.path.join(self.input_base, self.category)
        self.output_dir = os.path.join(self.output_base, self.category)
        self.discard_dir = os.path.join(self.discard_base, self.category)

        # Create folders
        for d in [self.output_dir, self.discard_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Get file list
        self.image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.png')]
        self.image_files.sort()
        self.index = 0

        # UI Components
        self.label_info = tk.Label(root, text="Loading...", font=("Arial", 14))
        self.label_info.pack(pady=10)

        self.canvas = tk.Canvas(root, width=800, height=800, bg="gray")
        self.canvas.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=20)

        self.btn_discard = tk.Button(self.btn_frame, text="DISCARD (X)", command=self.discard_current, bg="#ff4d4d", width=15, height=2)
        self.btn_discard.pack(side=tk.LEFT, padx=20)

        self.btn_keep = tk.Button(self.btn_frame, text="KEEP (Right)", command=self.keep_current, bg="#4CAF50", width=15, height=2)
        self.btn_keep.pack(side=tk.LEFT, padx=20)

        # Bind Keys
        self.root.bind("<Right>", lambda e: self.keep_current())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("x", lambda e: self.discard_current())
        self.root.bind("<Delete>", lambda e: self.discard_current())

        if not self.image_files:
            messagebox.showinfo("Done", f"No images found in {self.input_dir}")
            self.root.destroy()
            return

        self.show_image()

    def show_image(self):
        if self.index >= len(self.image_files):
            messagebox.showinfo("Done", "You have finished cleaning the folder!")
            self.root.destroy()
            return

        filename = self.image_files[self.index]
        file_path = os.path.join(self.input_dir, filename)
        
        self.label_info.config(text=f"[{self.index + 1}/{len(self.image_files)}] {filename}\nKeys: Right=Keep, X=Discard, Left=Back")

        # Load 16-bit PNG and convert to 8-bit for Display ONLY
        # Standard Tkinter/PIL might struggle with raw 16-bit display logic
        img_16 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img_16 is None:
            self.index += 1
            self.show_image()
            return

        # Simple 8-bit conversion for preview
        img_8 = (img_16 / 256).astype(np.uint8)
        
        # Resize to fit window while maintaining aspect ratio
        h, w = img_8.shape
        scale = min(800/w, 800/h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_8, (new_w, new_h))

        img_pil = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(400, 400, anchor=tk.CENTER, image=self.tk_img)

    def keep_current(self):
        filename = self.image_files[self.index]
        src = os.path.join(self.input_dir, filename)
        dst = os.path.join(self.output_dir, filename)
        
        shutil.move(src, dst)
        self.image_files.pop(self.index)
        self.show_image()

    def discard_current(self):
        filename = self.image_files[self.index]
        src = os.path.join(self.input_dir, filename)
        dst = os.path.join(self.discard_dir, filename)
        
        shutil.move(src, dst)
        self.image_files.pop(self.index)
        self.show_image()

    def prev_image(self):
        # Note: Since we are moving files out, 'back' only works if we stop moving 
        # but the request was specifically for a sorting application.
        # To keep it simple, back is disabled for 'moved' files.
        if self.index > 0:
            self.index -= 1
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = FabellaCleaner(root)
    root.mainloop()
