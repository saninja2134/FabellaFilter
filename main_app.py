# Main application module that integrates all tools into a single UI.
import os
# Suppress verbose warnings from TensorFlow and Albumentations before any imports
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys

# Add the parent directory to sys.path to allow importing from UI package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicom_converter import DicomConverter
from dataset_cleaner import FabellaCleaner
from obb_labeler import OBBLabeler
from seg_labeler import SegLabeler
from yolo_preparer import YoloPreparer
from model_trainer import ModelTrainer, ARCHITECTURES
from model_tester import ModelTester
from dataset_generator import DatasetGeneratorModal

# Colors
BG_COLOR = "#1E1E1E"
FG_COLOR = "#E0E0E0"
BUTTON_BG = "#333333"
BUTTON_ACTIVE = "#444444"
ACCENT_COLOR = "#007ACC"  # VS Code Blue for accent


class PrepareProgressWindow(tk.Toplevel):
    # Modal progress window shown during dataset preparation.
    def __init__(self, parent, title="Preparing Dataset..."):
        super().__init__(parent)
        self.title(title)
        self.geometry("560x380")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)
        self.grab_set()  # modal

        # Phase label
        self.phase_var = tk.StringVar(value="Starting...")
        tk.Label(self, textvariable=self.phase_var,
                 font=("Segoe UI", 11, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(
            padx=20, pady=(18, 4), anchor=tk.W)

        # Progress bar
        self.pct_var = tk.IntVar(value=0)
        self.bar = ttk.Progressbar(self, variable=self.pct_var, maximum=100,
                                   mode="determinate", length=520)
        self.bar.pack(padx=20, pady=(0, 4))

        self.pct_label_var = tk.StringVar(value="0%")
        tk.Label(self, textvariable=self.pct_label_var,
                 font=("Segoe UI", 9), bg=BG_COLOR, fg="#888888").pack(anchor=tk.E, padx=20)

        # Log area
        log_frame = tk.Frame(self, bg=BG_COLOR)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(6, 0))
        self._log_text = tk.Text(log_frame, height=10, bg="#252526", fg="#CCCCCC",
                                 font=("Consolas", 8), state=tk.DISABLED)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.config(yscrollcommand=sb.set)

        # Close button (disabled until done)
        self.close_btn = tk.Button(self, text="Close", state=tk.DISABLED,
                                   bg=BUTTON_BG, fg="white", relief=tk.FLAT,
                                   font=("Segoe UI", 10), cursor="hand2",
                                   command=self.destroy)
        self.close_btn.pack(pady=12)

    def update_step(self, pct, label):
        # Called from the worker thread via root.after.
        self.pct_var.set(pct)
        self.pct_label_var.set(f"{pct}%")
        if label:
            self.phase_var.set(label)
        if pct >= 100:
            self.close_btn.config(state=tk.NORMAL)
            self.phase_var.set("Complete!")

    def append_log(self, msg):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

class FabellaApp:
    # The main application class for the Fabella Dataset Manager.
    def __init__(self, root):
        # Initializes the FabellaApp.
        # Args:
        # root (tk.Tk): The root Tkinter window.
        self.root = root
        self.root.title("Fabella Dataset Manager")
        self.root.geometry("600x600")
        self.root.configure(bg=BG_COLOR)
        
        # Header
        self.header_frame = tk.Frame(root, bg=BG_COLOR)
        self.header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.title_label = tk.Label(
            self.header_frame, 
            text="Fabella Dataset Manager", 
            font=("Segoe UI", 20, "bold"), 
            bg=BG_COLOR, 
            fg=FG_COLOR
        )
        self.title_label.pack()

        self.subtitle_label = tk.Label(
            self.header_frame, 
            text="Central Hub for Dataset Processing & Training", 
            font=("Segoe UI", 10), 
            bg=BG_COLOR, 
            fg="#AAAAAA"
        )
        self.subtitle_label.pack(pady=(5, 0))

        # Main Content - Categories
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Style for Notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background="#2D2D2D", foreground="white", padding=[15, 5], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)], foreground=[("selected", "white")])
        style.configure("TFrame", background=BG_COLOR)

        # Tabs
        self.tab_dataset = ttk.Frame(self.notebook)
        self.tab_model = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dataset, text="Dataset Tools")
        self.notebook.add(self.tab_model, text="Model Training")
        
        self.setup_dataset_tab()
        self.setup_model_tab()
        
        # Log Area
        self.log_frame = tk.Frame(root, bg=BG_COLOR)
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        self.log_text = tk.Text(self.log_frame, height=8, bg="#252526", fg="#CCCCCC", font=("Consolas", 9))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.scrollbar.set)

        # Footer / Status
        self.status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#252526", fg="#CCCCCC")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def log(self, message):
        # Appends a message to the log text area.
        self.root.after(0, self.update_log_gui, message)

    def update_log_gui(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def set_status(self, message):
        # Safely updates the status bar text from any thread.
        self.root.after(0, lambda: self.status_bar.config(text=message))

    def heading(self, parent, text):
        # Creates a heading label.
        return tk.Label(parent, text=text, font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)

    def create_action_button(self, parent, text, description, command, color=BUTTON_BG):
        # Creates an action button with a description.
        frame = tk.Frame(parent, bg=BG_COLOR, pady=5)
        frame.pack(fill=tk.X)
        
        btn = tk.Button(
            frame,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg="white",
            activebackground=BUTTON_ACTIVE,
            activeforeground="white",
            relief=tk.FLAT,
            command=command,
            width=20,
            cursor="hand2"
        )
        btn.pack(side=tk.LEFT, padx=(0, 15))
        
        desc = tk.Label(
            frame,
            text=description,
            font=("Segoe UI", 9),
            bg=BG_COLOR,
            fg="#CCCCCC",
            wraplength=350,
            justify=tk.LEFT
        )
        desc.pack(side=tk.LEFT, fill=tk.X)
        return btn

    def setup_dataset_tab(self):
        # Sets up the Dataset Tools tab.
        container = tk.Frame(self.tab_dataset, bg=BG_COLOR, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        self.heading(container, "1. Pre-Processing").pack(anchor=tk.W, pady=(0, 10))
        
        self.create_action_button(
            container, 
            "Convert DICOM", 
            "Converts raw DICOM files to PNG format for processing.", 
            self.run_dicom_conversion
        )

        self.create_action_button(
            container, 
            "Clean Dataset", 
            "Sort positive PNG images into 'Sorted/pos' (Keep) and 'Discarded/pos' folders.", 
            self.run_cleaner
        )

        self.create_action_button(
            container, 
            "Sort Negatives", 
            "Review and sort negative PNG images into 'Sorted/neg' and 'Discarded/neg' folders.", 
            self.run_neg_sorter
        )

        self.heading(container, "2. Labeling").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container, 
            "Label OBB", 
            "Open the OBB Labeler tool to annotate sorted images with oriented boxes.", 
            self.run_labeler
        )

        self.create_action_button(
            container, 
            "Label Segmentation", 
            "Open the Segmentation Labeler tool to draw polygon masks.", 
            self.run_seg_labeler
        )
        
        self.heading(container, "3. Finalize").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container,
            "Prepare Dataset",
            "Open augmentation modal, then build Train/Val split and generate config for the selected architecture.",
            self.run_prepare_dataset
        )

    def setup_model_tab(self):
        # Sets up the Model Training tab.
        container = tk.Frame(self.tab_model, bg=BG_COLOR, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        self.heading(container, "4. Model Settings").pack(anchor=tk.W, pady=(0, 10))

        settings_frame = tk.Frame(container, bg=BG_COLOR)
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        # ── Row 0: Architecture ──────────────────────────────────
        tk.Label(settings_frame, text="Architecture:", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.arch_var = tk.StringVar(value="YOLO Seg")
        arch_names = list(ARCHITECTURES.keys())
        self.arch_cb = ttk.Combobox(settings_frame, textvariable=self.arch_var,
                                    values=arch_names, state="readonly", width=14)
        self.arch_cb.grid(row=0, column=1, padx=5, pady=5)
        self.arch_cb.bind("<<ComboboxSelected>>", self._on_arch_change)

        # ── Row 0: Version (hidden for RT-DETR / RF-DETR) ────────
        self.version_label = tk.Label(settings_frame, text="Version:", bg=BG_COLOR, fg=FG_COLOR)
        self.version_label.grid(row=0, column=2, padx=(20, 5), pady=5, sticky=tk.W)
        self.version_var = tk.StringVar(value="11")
        self.version_cb = ttk.Combobox(settings_frame, textvariable=self.version_var,
                                       values=["8","9","10","11","12","26"], width=5)
        self.version_cb.grid(row=0, column=3, padx=5, pady=5)

        # ── Row 1: Size ──────────────────────────────────────────
        tk.Label(settings_frame, text="Size:", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.size_var = tk.StringVar(value="n")
        self.size_cb = ttk.Combobox(settings_frame, textvariable=self.size_var,
                                    values=["n","s","m","l","x"], state="readonly", width=8)
        self.size_cb.grid(row=1, column=1, padx=5, pady=5)

        # ── Row 1: Epochs / Batch ────────────────────────────────
        tk.Label(settings_frame, text="Epochs:", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=1, column=2, padx=(20, 5), pady=5, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="100")
        tk.Entry(settings_frame, textvariable=self.epochs_var, width=6,
                 bg="#2D2D2D", fg=FG_COLOR, insertbackground=FG_COLOR).grid(
            row=1, column=3, padx=5, pady=5)

        tk.Label(settings_frame, text="Batch:", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=1, column=4, padx=(15, 5), pady=5, sticky=tk.W)
        self.batch_var = tk.StringVar(value="4")
        tk.Entry(settings_frame, textvariable=self.batch_var, width=5,
                 bg="#2D2D2D", fg=FG_COLOR, insertbackground=FG_COLOR).grid(
            row=1, column=5, padx=5, pady=5)

        # ── Row 2: Img size ─────────────────────────────────────
        tk.Label(settings_frame, text="Img Size:", bg=BG_COLOR, fg=FG_COLOR).grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.imgsz_var = tk.StringVar(value="1024")
        tk.Entry(settings_frame, textvariable=self.imgsz_var, width=7,
                 bg="#2D2D2D", fg=FG_COLOR, insertbackground=FG_COLOR).grid(
            row=2, column=1, padx=5, pady=5)

        self.heading(container, "5. Training").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container,
            "Train Model",
            "Train the selected architecture on the prepared dataset (GPU recommended).",
            self.run_train_model,
            color="#2E7D32"
        )

        self.heading(container, "6. Evaluation").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container,
            "Test Model",
            "Run inference on unsorted images and sort results into detected/undetected.",
            self.run_test_model,
            color="#C62828"
        )

    def run_in_thread(self, target, status_msg):
        # Runs a target function in a separate thread to keep UI responsive.
        def wrapper():
            self.set_status(status_msg)
            try:
                target()
                self.set_status("Ready")
            except Exception as e:
                self.log(f"Error: {e}")
                self.set_status("Error occurred")
                messagebox.showerror("Error", str(e))

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    def run_dicom_conversion(self):
        # Runs the DICOM to PNG conversion process.
        converter = DicomConverter()
        self.run_in_thread(lambda: converter.run_conversion(progress_callback=self.log), "Converting DICOM files...")

    def run_cleaner(self):
        # Opens the Dataset Cleaner window for positive images.
        self.set_status("Opening Dataset Cleaner...")
        cleaner_window = FabellaCleaner(self.root)
        self.set_status("Ready")

    def run_neg_sorter(self):
        # Opens the Dataset Cleaner window scoped to negative images.
        # "Keep" = truly negative (no fabella) → data/sorted/neg
        # "Discard" = actually has a fabella → data/sorted/pos
        self.set_status("Opening Negative Sorter...")
        FabellaCleaner(
            self.root,
            category="neg",
            discard_dir_override="data/sorted/pos",
            keep_label="NO FABELLA →",
            discard_label="← HAS FABELLA",
            window_title="Negative Sorter — Has Fabella? Move to Pos"
        )
        self.set_status("Ready")

    def run_labeler(self):
        # Runs the OBB Labeler tool.
        self.set_status("Opening OBB Labeler...")
        labeler = OBBLabeler()
        # Run in main thread as it uses OpenCV GUI which needs to be in main thread
        try:
            labeler.run()
            self.set_status("Ready")
        except Exception as e:
            self.log(f"Error: {e}")
            self.set_status("Error occurred")
            messagebox.showerror("Error", str(e))

    def run_seg_labeler(self):
        # Runs the Segmentation Labeler tool.
        self.set_status("Opening Segmentation Labeler...")
        labeler = SegLabeler()
        try:
            labeler.run()
            self.set_status("Ready")
        except Exception as e:
            self.log(f"Error: {e}")
            self.set_status("Error occurred")
            messagebox.showerror("Error", str(e))
        
    def _on_arch_change(self, _event=None):
        # Update version and size dropdowns when architecture changes.
        arch = self.arch_var.get()
        info = ARCHITECTURES.get(arch, {})
        versions = info.get("versions", [""])
        sizes    = info.get("sizes", ["n"])

        # Version dropdown — hide if not applicable
        if versions == [""]:
            self.version_label.grid_remove()
            self.version_cb.grid_remove()
        else:
            self.version_label.grid()
            self.version_cb.grid()
            self.version_cb.config(values=versions)
            self.version_var.set(versions[-1] if versions else "")

        self.size_cb.config(values=sizes)
        self.size_var.set(sizes[0])

    def _get_arch_info(self):
        # Returns (arch, version, size, epochs, imgsz, batch) from current UI state.
        arch    = self.arch_var.get()
        version = self.version_var.get()
        size    = self.size_var.get()
        try:
            epochs = int(self.epochs_var.get())
        except ValueError:
            epochs = 100
        try:
            imgsz = int(self.imgsz_var.get())
        except ValueError:
            imgsz = 1024
        try:
            batch = int(self.batch_var.get())
        except ValueError:
            batch = 4
        return arch, version, size, epochs, imgsz, batch

    def _get_yolo_task(self):
        # Maps current architecture to task key used by YoloPreparer.
        arch = self.arch_var.get()
        if arch == "YOLO OBB":
            return "obb"
        if arch == "RT-DETR":
            return "detect"
        return "segment"  # YOLO Seg default

    def run_prepare_dataset(self):
        # Opens the augmentation modal, then shows a progress window while preparing.
        arch = self.arch_var.get()
        task = self._get_yolo_task()
        also_coco = (ARCHITECTURES.get(arch, {}).get("format") == "coco")

        def on_generate(config):
            prog = PrepareProgressWindow(self.root, title=f"Preparing Dataset — {arch}")

            def step_cb(pct, label):
                self.root.after(0, prog.update_step, pct, label)

            def log_cb(msg):
                self.root.after(0, self.update_log_gui, msg)
                self.root.after(0, prog.append_log, msg)

            def run():
                self.set_status(f"Preparing dataset ({arch})...")
                try:
                    preparer = YoloPreparer(task=task)
                    preparer.setup_dataset(
                        config=config,
                        also_export_coco=also_coco,
                        progress_callback=log_cb,
                        step_callback=step_cb,
                    )
                    self.set_status("Ready")
                except Exception as e:
                    self.root.after(0, self.log, f"Error: {e}")
                    self.set_status("Error occurred")
                    self.root.after(0, messagebox.showerror, "Error", str(e))

            t = threading.Thread(target=run, daemon=True)
            t.start()

        DatasetGeneratorModal(self.root, task=task, on_generate=on_generate)

    def run_train_model(self):
        # Trains the selected architecture.
        arch, version, size, epochs, imgsz, batch = self._get_arch_info()
        trainer = ModelTrainer(arch=arch, version=version, size=size,
                               epochs=epochs, imgsz=imgsz, batch=batch)

        def do_train():
            trainer.train(progress_callback=self.log)
            path = getattr(trainer, 'results_plot_path', None)
            if path and os.path.exists(path):
                self.root.after(0, lambda: self._show_results_plot(path))

        self.run_in_thread(do_train, f"Training {arch} {size}...")

    def _show_results_plot(self, path):
        """Popup window displaying the training results chart."""
        try:
            from PIL import Image, ImageTk
            win = tk.Toplevel(self.root)
            win.title("Training Results")
            win.resizable(True, True)

            img = Image.open(path)
            img.thumbnail((1200, 700), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            lbl = tk.Label(win, image=photo)
            lbl.image = photo   # keep reference
            lbl.pack(padx=10, pady=10)

            tk.Button(win, text="Close", command=win.destroy,
                      width=14).pack(pady=(0, 10))
            win.grab_set()
        except Exception as e:
            self.log(f"Could not show results plot: {e}")

    def run_test_model(self):
        # Runs inference with the selected trained model.
        arch, _version, size, *_ = self._get_arch_info()
        tester = ModelTester(arch=arch, size=size)
        self.run_in_thread(
            lambda: tester.run_test(progress_callback=self.log),
            f"Testing {arch} {size}..."
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = FabellaApp(root)
    root.mainloop()
