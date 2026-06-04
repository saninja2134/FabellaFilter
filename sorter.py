# Module for cleaning and sorting the dataset.
import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from trainer import ModelRegistry
from classifier_utils import (
    CLASSIFIER_ARCH,
    DEFAULT_AUTO_POSITIVE_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    load_classifier_checkpoint,
    predict_fabella_probability,
)

# Colors
BG_COLOR = "#1E1E1E"
FG_COLOR = "#E0E0E0"
ACCENT_KEEP = "#4CAF50"  # Green
ACCENT_DISCARD = "#FF5252"  # Red
BUTTON_BG = "#333333"
BUTTON_ACTIVE = "#444444"
ACCENT_MODEL = "#1976D2"


class ClassifierTriageDialog(tk.Toplevel):
    def __init__(self, parent, models):
        super().__init__(parent)
        self.title("Classifier-Assisted Triage")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.models = models

        self.model_var = tk.StringVar()
        self.auto_var = tk.StringVar(value=f"{DEFAULT_AUTO_POSITIVE_THRESHOLD:.2f}")
        self.review_var = tk.StringVar(value=f"{DEFAULT_REVIEW_THRESHOLD:.2f}")

        container = tk.Frame(self, bg=BG_COLOR, padx=18, pady=16)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            container,
            text="Use a trained classifier to auto-promote only the safest fabella candidates.",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=FG_COLOR,
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor=tk.W, pady=(0, 12))

        tk.Label(container, text="Model:", bg=BG_COLOR, fg=FG_COLOR).pack(anchor=tk.W)
        self.model_cb = ttk.Combobox(
            container,
            textvariable=self.model_var,
            state="readonly",
            width=56,
        )
        self.model_labels = []
        for entry in models:
            self.model_labels.append(
                f"{entry.get('arch', CLASSIFIER_ARCH)} [{entry.get('size', '?')}] - "
                f"{entry.get('date', 'unknown')}"
            )
        self.model_cb.config(values=self.model_labels)
        self.model_cb.current(0)
        self.model_cb.pack(fill=tk.X, pady=(4, 12))

        thresh = tk.Frame(container, bg=BG_COLOR)
        thresh.pack(fill=tk.X, pady=(0, 10))

        tk.Label(thresh, text="Auto-positive >=", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4
        )
        tk.Entry(
            thresh,
            textvariable=self.auto_var,
            width=8,
            bg="#2D2D2D",
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
        ).grid(row=0, column=1, sticky=tk.W, pady=4)

        tk.Label(thresh, text="Review >=", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4
        )
        tk.Entry(
            thresh,
            textvariable=self.review_var,
            width=8,
            bg="#2D2D2D",
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
        ).grid(row=1, column=1, sticky=tk.W, pady=4)

        tk.Label(
            container,
            text=(
                "Images above the auto-positive threshold are moved into sorted/pos. "
                "Everything else stays in the current folder; the review-band images "
                "are simply shown first in the queue."
            ),
            font=("Segoe UI", 9),
            bg=BG_COLOR,
            fg="#999999",
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor=tk.W, pady=(0, 12))

        btns = tk.Frame(container, bg=BG_COLOR)
        btns.pack(fill=tk.X)
        tk.Button(
            btns,
            text="Cancel",
            command=self.destroy,
            bg=BUTTON_BG,
            fg=FG_COLOR,
            activebackground=BUTTON_ACTIVE,
            relief=tk.FLAT,
            padx=16,
        ).pack(side=tk.RIGHT)
        tk.Button(
            btns,
            text="Run Triage",
            command=self._on_confirm,
            bg=ACCENT_MODEL,
            fg="white",
            activebackground=ACCENT_MODEL,
            relief=tk.FLAT,
            padx=16,
        ).pack(side=tk.RIGHT, padx=(0, 8))

    def _on_confirm(self):
        try:
            auto = float(self.auto_var.get())
            review = float(self.review_var.get())
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Thresholds must be numeric values.", parent=self)
            return

        if not (0.0 <= review <= 1.0 and 0.0 <= auto <= 1.0):
            messagebox.showerror(
                "Invalid Threshold",
                "Thresholds must be between 0.0 and 1.0.",
                parent=self,
            )
            return
        if review > auto:
            messagebox.showerror(
                "Invalid Threshold",
                "Review threshold must be less than or equal to the auto-positive threshold.",
                parent=self,
            )
            return

        idx = self.model_cb.current()
        if idx < 0:
            idx = 0
        self.result = {
            "model_entry": self.models[idx],
            "auto_threshold": auto,
            "review_threshold": review,
        }
        self.destroy()

class FabellaCleaner(tk.Toplevel):
    # A Tkinter Toplevel window for sorting images into keep/discard categories.
    def __init__(self, parent, input_base="data/png", output_base="data/sorted", discard_base="data/discarded",
                 category="pos", discard_dir_override=None, keep_label="KEEP →", discard_label="← DISCARD",
                 window_title="Fabella Dataset Cleaner"):
        # Initializes the FabellaCleaner window.
        # 
        # Args:
        # parent (tk.Tk or tk.Toplevel): The parent window.
        # input_base (str): The base directory containing input images.
        # output_base (str): The base directory to save kept images.
        # discard_base (str): The base directory to save discarded images.
        # category (str): The category to process (e.g., 'pos' or 'neg').
        # discard_dir_override (str): If set, discarded images go here instead of discard_base/category.
        # keep_label (str): Text for the keep button.
        # discard_label (str): Text for the discard button.
        # window_title (str): Title of the window.
        super().__init__(parent)
        self.title(window_title)
        self.geometry("1000x900")
        self.configure(bg=BG_COLOR)

        # CONFIGURATION
        self.input_base = input_base
        self.output_base = output_base
        self.discard_base = discard_base
        self._keep_label = keep_label
        self._discard_label = discard_label
        
        self.category = category 
        self.input_dir = os.path.join(self.input_base, self.category)
        self.output_dir = os.path.join(self.output_base, self.category)
        if discard_dir_override:
            self.discard_dir = discard_dir_override
        else:
            self.discard_dir = os.path.join(self.discard_base, self.category)
        self.auto_positive_dir = os.path.join(self.output_base, "pos")

        for d in [self.output_dir, self.discard_dir, self.auto_positive_dir]:
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
        self.model_scores = {}
        self.triage_state = None

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
        
        self.update_idletasks() # Forces Tkinter to calculate true window dimensions
        self.show_image()

    def setup_ui(self):
        # Sets up the user interface elements.
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

        self.model_score_label = tk.Label(
            self,
            text="Model triage not run.",
            font=("Segoe UI", 9),
            bg=BG_COLOR,
            fg="#888888"
        )
        self.model_score_label.pack(pady=(0, 5))

        # Controls / Instructions
        self.controls_frame = tk.Frame(self, bg=BG_COLOR)
        self.controls_frame.pack(fill=tk.X, padx=50, pady=(10, 30))
        
        # Create custom styled buttons
        self.btn_discard = self.create_button(
            self.controls_frame, 
            self._discard_label, 
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

        self.btn_model_triage = self.create_button(
            self.controls_frame,
            "MODEL TRIAGE",
            ACCENT_MODEL,
            self.run_model_triage
        )
        self.btn_model_triage.pack(side=tk.LEFT, padx=10)

        self.btn_keep = self.create_button(
            self.controls_frame, 
            self._keep_label, 
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
        # Creates a styled button.
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
        # Binds keyboard and mouse events.
        self.bind("<Right>", lambda e: self.keep_current())
        self.bind("<Left>", lambda e: self.discard_current())
        self.bind("<Control-z>", lambda e: self.undo_last())
        self.bind("r", lambda e: self.reset_zoom())
        
        # Mouse bindings for Zoom/Pan
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)

    def update_status(self):
        # Updates the status labels.
        self.count_label.config(text=f"{self.processed_count} processed | {len(self.image_files)} remaining")
        
        if self.image_files:
            current_file = self.image_files[0]
            self.filename_label.config(text=current_file)
            self._update_model_score_label(current_file)
        else:
            self.filename_label.config(text="No more images")
            if self.triage_state:
                self.model_score_label.config(
                    text=f"Triage complete ({self.triage_state['run_name']}).",
                    fg="#888888",
                )
            else:
                self.model_score_label.config(text="Model triage not run.", fg="#888888")

    def show_image(self):
        # Displays the current image on the canvas.
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
        # Resets zoom and pan variables.
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.zoom_info_label.config(text="Zoom: 100%")

    def reset_zoom(self):
        # Resets the view to default zoom and pan.
        self.reset_zoom_vars()
        self.redraw_image()

    def zoom(self, event):
        # Handles mouse wheel zoom events.
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
        # Records the starting position for panning.
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        # Handles mouse drag panning events.
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
        # Redraws the image on the canvas with current zoom and pan.
        if self.original_image_cv is None:
            return
            
        if len(self.original_image_cv.shape) == 3:
            h, w, _ = self.original_image_cv.shape
        else:
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
        
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        
        # Center image + Pan offset
        center_x = (canvas_width // 2) + self.pan_x
        center_y = (canvas_height // 2) - 50 + self.pan_y # Offset for header
        
        self.canvas.delete("all")
        self.canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.tk_img)

    def move_file(self, destination):
        # Moves the current file to the specified destination.
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
            self.model_scores.pop(filename, None)
            self.show_image()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move file: {e}", parent=self)

    def keep_current(self):
        # Moves the current image to the keep directory.
        self.move_file(self.output_dir)

    def discard_current(self):
        # Moves the current image to the discard directory.
        self.move_file(self.discard_dir)
        
    def undo_last(self):
        # Undoes the last move operation.
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

    def _update_model_score_label(self, filename):
        score = self.model_scores.get(filename)
        if score is None:
            if self.triage_state:
                self.model_score_label.config(
                    text=(
                        f"Triage active ({self.triage_state['run_name']}) - "
                        "this file was not scored."
                    ),
                    fg="#888888",
                )
            else:
                self.model_score_label.config(text="Model triage not run.", fg="#888888")
            return

        band = "MANUAL"
        color = "#888888"
        auto_threshold = self.triage_state["auto_threshold"] if self.triage_state else 1.0
        review_threshold = self.triage_state["review_threshold"] if self.triage_state else 1.0
        if score >= auto_threshold:
            band = "AUTO-POSITIVE"
            color = ACCENT_KEEP
        elif score >= review_threshold:
            band = "REVIEW BAND"
            color = "#FFB74D"

        self.model_score_label.config(
            text=f"Model fabella score: {score * 100:.1f}% - {band}",
            fg=color,
        )

    def run_model_triage(self):
        if not self.image_files:
            messagebox.showinfo("No Images", "There are no remaining images to triage.", parent=self)
            return

        classifier_models = [
            entry for entry in ModelRegistry.all_models()
            if entry.get("arch") == CLASSIFIER_ARCH and os.path.exists(entry.get("weights_path", ""))
        ]
        if not classifier_models:
            messagebox.showwarning(
                "No Classifier Models",
                "Train a Torchvision Classifier first, then come back to use model triage.",
                parent=self,
            )
            return

        dialog = ClassifierTriageDialog(self, classifier_models)
        self.wait_window(dialog)
        if not dialog.result:
            return

        entry = dialog.result["model_entry"]
        auto_threshold = dialog.result["auto_threshold"]
        review_threshold = dialog.result["review_threshold"]
        weights_path = entry.get("weights_path", "")
        if not os.path.exists(weights_path):
            messagebox.showerror(
                "Missing Weights",
                f"Model weights were not found at:\n{weights_path}",
                parent=self,
            )
            return

        try:
            model, checkpoint, device, transform = load_classifier_checkpoint(weights_path)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not load classifier:\n{exc}", parent=self)
            return

        self.config(cursor="watch")
        self.count_label.config(text="Running model triage...")
        self.update_idletasks()

        moved_auto = 0
        review_files = []
        manual_files = []
        scored_files = list(self.image_files)

        try:
            for idx, filename in enumerate(scored_files, start=1):
                src = os.path.join(self.input_dir, filename)
                if not os.path.exists(src):
                    continue

                try:
                    prob = predict_fabella_probability(model, transform, src, device)
                except Exception as exc:
                    print(f"Triage error on {filename}: {exc}")
                    prob = 0.0

                self.model_scores[filename] = prob
                if prob >= auto_threshold:
                    dst = os.path.join(self.auto_positive_dir, filename)
                    shutil.move(src, dst)
                    self.history.append((filename, self.input_dir, self.auto_positive_dir))
                    moved_auto += 1
                    self.processed_count += 1
                elif prob >= review_threshold:
                    review_files.append(filename)
                else:
                    manual_files.append(filename)

                if idx % 10 == 0 or idx == len(scored_files):
                    self.count_label.config(
                        text=f"Model triage {idx}/{len(scored_files)} - auto+ {moved_auto}"
                    )
                    self.update_idletasks()
        finally:
            self.config(cursor="")

        self.triage_state = {
            "run_name": entry.get("run_name", "classifier"),
            "auto_threshold": auto_threshold,
            "review_threshold": review_threshold,
            "checkpoint_thresholds": {
                "auto": checkpoint.get("auto_positive_threshold"),
                "review": checkpoint.get("review_threshold"),
            },
        }

        self.image_files = review_files + manual_files
        self.total_images = self.processed_count + len(self.image_files)

        if self.image_files:
            self.show_image()
        else:
            self.update_status()
            self.canvas.delete("all")
            self.filename_label.config(text="No more images")
            self.model_score_label.config(
                text=f"Triage complete ({self.triage_state['run_name']}).",
                fg="#888888",
            )

        messagebox.showinfo(
            "Model Triage Complete",
            (
                f"Auto-moved to sorted/pos: {moved_auto}\n"
                f"Review-band queued first: {len(review_files)}\n"
                f"Remaining manual after review band: {len(manual_files)}"
            ),
            parent=self,
        )

        if not self.image_files:
            self.destroy()
