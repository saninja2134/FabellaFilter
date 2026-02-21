"""
Module for cleaning and sorting the dataset.
"""
import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Colors
BG_COLOR = "#1E1E1E"
FG_COLOR = "#E0E0E0"
ACCENT_KEEP = "#4CAF50"  # Green
ACCENT_DISCARD = "#FF5252"  # Red
BUTTON_BG = "#333333"
BUTTON_ACTIVE = "#444444"

class FabellaCleaner(tk.Toplevel):
    """
    A Tkinter Toplevel window for sorting images into keep/discard categories.
    """
    def __init__(self, parent, input_base="dataset_png", output_base="dataset_sorted", discard_base="dataset_discarded", category="pos"):
        """
        Initializes the FabellaCleaner window.
        
        Args:
            parent (tk.Tk or tk.Toplevel): The parent window.
            input_base (str): The base directory containing input images.
            output_base (str): The base directory to save kept images.
            discard_base (str): The base directory to save discarded images.
            category (str): The category to process (e.g., 'pos' or 'neg').
        """
        super().__init__(parent)
        self.title("Fabella Dataset Cleaner")
        self.geometry("1000x900")
        self.configure(bg=BG_COLOR)

        # CONFIGURATION
        self.input_base = input_base
        self.output_base = output_base
        self.discard_base = discard_base
        
        self.category = category 
        self.input_dir = os.path.join(self.input_base, self.category)
        self.output_dir = os.path.join(self.output_base, self.category)
        self.discard_dir = os.path.join(self.discard_base, self.category)

        for d in [self.output_dir, self.discard_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Load files
        if os.path.exists(self.input_dir):
            self.image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.png')]
            self.image_files.sort()
        else:
            self.image_files = []
        
        # History for Undo
        self.history = []
        
        self.total_images = len(self.image_files)
        self.processed_count = 0

        # Zoom and Pan variables
        self.current_image = None
        self.original_image_cv = None
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.setup_ui()
        self.bind_keys()
        
        if not self.image_files:
            messagebox.showinfo("Done", f"No images found in {self.input_dir}", parent=self)
            self.destroy()
            return
            
        self.show_image()

    def setup_ui(self):
        """Sets up the user interface elements."""
        # Header / Status
        self.header_frame = tk.Frame(self, bg=BG_COLOR)
        self.header_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.title_label = tk.Label(
            self.header_frame, 
            text="Dataset Cleaner", 
            font=("Segoe UI", 18, "bold"), 
            bg=BG_COLOR, 
            fg=FG_COLOR
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.count_label = tk.Label(
            self.header_frame, 
            text="0 / 0", 
            font=("Segoe UI", 12), 
            bg=BG_COLOR, 
            fg="#AAAAAA"
        )
        self.count_label.pack(side=tk.RIGHT)

        # Main Image Area
        self.canvas_frame = tk.Frame(self, bg=BG_COLOR)
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        self.canvas_border = tk.Frame(self.canvas_frame, bg="#333333", padx=2, pady=2)
        self.canvas_border.pack(expand=True, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(
            self.canvas_border, 
            bg="#000000", 
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Filename Label
        self.filename_label = tk.Label(
            self,
            text="", 
            font=("Consolas", 10), 
            bg=BG_COLOR, 
            fg="#888888"
        )
        self.filename_label.pack(pady=(0, 2))
        
        self.zoom_info_label = tk.Label(
            self,
            text="Zoom: 100%", 
            font=("Segoe UI", 9), 
            bg=BG_COLOR, 
            fg="#666666"
        )
        self.zoom_info_label.pack(pady=(0, 5))

        # Controls / Instructions
        self.controls_frame = tk.Frame(self, bg=BG_COLOR)
        self.controls_frame.pack(fill=tk.X, padx=50, pady=(10, 30))
        
        # Create custom styled buttons
        self.btn_discard = self.create_button(
            self.controls_frame, 
            "← DISCARD", 
            ACCENT_DISCARD, 
            self.discard_current
        )
        self.btn_discard.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.btn_reset_zoom = self.create_button(
            self.controls_frame, 
            "RESET VIEW", 
            "#555555", 
            self.reset_zoom
        )
        self.btn_reset_zoom.pack(side=tk.LEFT, padx=10)

        self.btn_keep = self.create_button(
            self.controls_frame, 
            "KEEP →", 
            ACCENT_KEEP, 
            self.keep_current
        )
        self.btn_keep.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        
        # Instructions footer
        self.footer_label = tk.Label(
            self,
            text="Shortcuts: Left=Discard | Right=Keep | Ctrl+Z=Undo | Wheel=Zoom | Drag=Pan",
            font=("Segoe UI", 9),
            bg=BG_COLOR,
            fg="#666666"
        )
        self.footer_label.pack(pady=(0, 15))

    def create_button(self, parent, text, bg_color, command):
        """Creates a styled button."""
        btn = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 12, "bold"),
            bg=bg_color,
            fg="white",
            activebackground=bg_color,
            activeforeground="white",
            relief=tk.FLAT,
            command=command,
            cursor="hand2",
            height=2
        )
        return btn

    def bind_keys(self):
        """Binds keyboard and mouse events."""
        self.bind("<Right>", lambda e: self.keep_current())
        self.bind("<Left>", lambda e: self.discard_current())
        self.bind("<Control-z>", lambda e: self.undo_last())
        self.bind("r", lambda e: self.reset_zoom())
        
        # Mouse bindings for Zoom/Pan
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)

    def update_status(self):
        """Updates the status labels."""
        self.count_label.config(text=f"{self.processed_count} processed | {len(self.image_files)} remaining")
        
        if self.image_files:
            current_file = self.image_files[0]
            self.filename_label.config(text=current_file)
        else:
            self.filename_label.config(text="No more images")

    def show_image(self):
        """Displays the current image on the canvas."""
        if not self.image_files:
            messagebox.showinfo("Complete", "All images have been sorted!", parent=self)
            self.destroy()
            return

        self.update_status()
        self.reset_zoom_vars()
        
        filename = self.image_files[0]
        file_path = os.path.join(self.input_dir, filename)
        
        try:
            img_16 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img_16 is None:
                print(f"Error loading {filename}, skipping.")
                self.image_files.pop(0)
                self.show_image()
                return

            self.original_image_cv = (img_16 / 256).astype(np.uint8)
            self.redraw_image()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            self.image_files.pop(0)
            self.show_image()

    def reset_zoom_vars(self):
        """Resets zoom and pan variables."""
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.zoom_info_label.config(text="Zoom: 100%")

    def reset_zoom(self):
        """Resets the view to default zoom and pan."""
        self.reset_zoom_vars()
        self.redraw_image()

    def zoom(self, event):
        """Handles mouse wheel zoom events."""
        if self.original_image_cv is None:
            return
            
        # Zoom factor
        scale = 1.1 if event.delta > 0 else 0.9
        new_scale = self.zoom_scale * scale
        
        # Limit zoom
        if 0.1 < new_scale < 10.0:
            self.zoom_scale = new_scale
            self.zoom_info_label.config(text=f"Zoom: {int(self.zoom_scale * 100)}%")
            self.redraw_image()

    def start_pan(self, event):
        """Records the starting position for panning."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        """Handles mouse drag panning events."""
        if self.original_image_cv is None:
            return
            
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        self.pan_x += dx
        self.pan_y += dy
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        self.redraw_image()

    def redraw_image(self):
        """Redraws the image on the canvas with current zoom and pan."""
        if self.original_image_cv is None:
            return
            
        h, w = self.original_image_cv.shape
        
        # Calculate fit scale first so image fits in window by default
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        # Fallback if window not fully drawn
        if canvas_width < 100: canvas_width = 800
        if canvas_height < 100: canvas_height = 800
        
        scale_fit = min((canvas_width-40)/w, (canvas_height-200)/h) # Subtract padding
        
        # Apply zoom
        final_scale = scale_fit * self.zoom_scale
        
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        if new_w <= 0 or new_h <= 0: return

        # Resize image
        img_resized = cv2.resize(self.original_image_cv, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        
        # Center image + Pan offset
        center_x = (canvas_width // 2) + self.pan_x
        center_y = (canvas_height // 2) - 50 + self.pan_y # Offset for header
        
        self.canvas.delete("all")
        self.canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.tk_img)

    def move_file(self, destination):
        """Moves the current file to the specified destination."""
        if not self.image_files:
            return
            
        filename = self.image_files[0]
        src = os.path.join(self.input_dir, filename)
        dst = os.path.join(destination, filename)
        
        try:
            shutil.move(src, dst)
            self.history.append((filename, self.input_dir, destination))
            self.image_files.pop(0)
            self.processed_count += 1
            self.show_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move file: {e}", parent=self)

    def keep_current(self):
        """Moves the current image to the keep directory."""
        self.move_file(self.output_dir)

    def discard_current(self):
        """Moves the current image to the discard directory."""
        self.move_file(self.discard_dir)
        
    def undo_last(self):
        """Undoes the last move operation."""
        if not self.history:
            return
            
        filename, original_src, current_loc = self.history.pop()
        
        src = os.path.join(current_loc, filename)
        dst = os.path.join(original_src, filename)
        
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                self.image_files.insert(0, filename)
                self.processed_count -= 1
                self.show_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to undo: {e}", parent=self)
        else:
            messagebox.showwarning("Warning", f"File {filename} not found in {src}", parent=self)
