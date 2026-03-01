# Main application module that integrates all tools into a single UI.
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add the parent directory to sys.path to allow importing from UI package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dicom_converter import DicomConverter
from dataset_cleaner import FabellaCleaner
from obb_labeler import OBBLabeler
from seg_labeler import SegLabeler
from yolo_preparer import YoloPreparer
from yolo_trainer import YoloTrainer
from yolo_tester import YoloTester
from dataset_generator import DatasetGeneratorModal

# Colors
BG_COLOR = "#1E1E1E"
FG_COLOR = "#E0E0E0"
BUTTON_BG = "#333333"
BUTTON_ACTIVE = "#444444"
ACCENT_COLOR = "#007ACC"  # VS Code Blue for accent

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
            "Prepare YOLO", 
            "Split dataset into Train/Val/Test and generate config.", 
            self.run_prepare_yolo
        )

    def setup_model_tab(self):
        # Sets up the Model Training tab.
        container = tk.Frame(self.tab_model, bg=BG_COLOR, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        self.heading(container, "4. Model Settings").pack(anchor=tk.W, pady=(0, 10))
        
        settings_frame = tk.Frame(container, bg=BG_COLOR)
        settings_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(settings_frame, text="Task Type:", bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.task_var = tk.StringVar(value="Segmentation")
        task_cb = ttk.Combobox(settings_frame, textvariable=self.task_var, values=["Segmentation", "OBB"], state="readonly", width=15)
        task_cb.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(settings_frame, text="YOLO Version:", bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=2, padx=(20, 5), pady=5, sticky=tk.W)
        self.version_var = tk.StringVar(value="11")
        version_cb = ttk.Combobox(settings_frame, textvariable=self.version_var, values=["8", "9", "10", "11", "12", "26"], width=5)
        version_cb.grid(row=0, column=3, padx=5, pady=5)

        self.heading(container, "5. Training").pack(anchor=tk.W, pady=(0, 10))
        
        self.create_action_button(
            container, 
            "Train Model", 
            "Start training the selected YOLO model on the prepared dataset (GPU recommended).", 
            self.run_train_yolo,
            color="#2E7D32"  # Greenish
        )

        self.heading(container, "5. Evaluation").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container, 
            "Test Model", 
            "Run inference on the test set and visualize results.", 
            self.run_test_model,
            color="#C62828"  # Reddish
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
        
    def get_current_task(self):
        return "obb" if self.task_var.get() == "OBB" else "segment"

    def run_prepare_yolo(self):
        # Opens the Dataset Generation & Augmentation modal, then runs YOLO preparation
        # with the returned config when the user clicks Generate.
        task = self.get_current_task()

        def on_generate(config):
            preparer = YoloPreparer(task=task)
            self.run_in_thread(
                lambda: preparer.setup_dataset(config=config, progress_callback=self.log),
                f"Preparing {task.upper()} dataset..."
            )

        DatasetGeneratorModal(self.root, task=task, on_generate=on_generate)

    def run_train_yolo(self):
        # Runs the YOLO model training process.
        task = self.get_current_task()
        version = self.version_var.get()
        trainer = YoloTrainer(task=task, model_version=version)
        self.run_in_thread(lambda: trainer.train_fabella(progress_callback=self.log), f"Training YOLOv{version} {task.upper()} model...")

    def run_test_model(self):
        # Runs the YOLO model testing process.
        task = self.get_current_task()
        tester = YoloTester(task=task)
        self.run_in_thread(lambda: tester.run_test(progress_callback=self.log), f"Testing {task.upper()} model...")

if __name__ == "__main__":
    root = tk.Tk()
    app = FabellaApp(root)
    root.mainloop()
