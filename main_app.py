"""
Main application module that integrates all tools into a single UI.
"""
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
from yolo_preparer import YoloPreparer
from yolo_trainer import YoloTrainer
from yolo_tester import YoloTester

# Colors
BG_COLOR = "#1E1E1E"
FG_COLOR = "#E0E0E0"
BUTTON_BG = "#333333"
BUTTON_ACTIVE = "#444444"
ACCENT_COLOR = "#007ACC"  # VS Code Blue for accent

class FabellaApp:
    """
    The main application class for the Fabella Dataset Manager.
    """
    def __init__(self, root):
        """
        Initializes the FabellaApp.
        
        Args:
            root (tk.Tk): The root Tkinter window.
        """
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
        """Appends a message to the log text area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def set_status(self, message):
        """Updates the status bar text."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def heading(self, parent, text):
        """Creates a heading label."""
        return tk.Label(parent, text=text, font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)

    def create_action_button(self, parent, text, description, command, color=BUTTON_BG):
        """Creates an action button with a description."""
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
        """Sets up the Dataset Tools tab."""
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
            "Sort images into 'Sorted' (Keep) and 'Discarded' folders.", 
            self.run_cleaner
        )

        self.heading(container, "2. Labeling").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container, 
            "Label Images", 
            "Open the OBB Labeler tool to annotate sorted images.", 
            self.run_labeler
        )
        
        self.heading(container, "3. Finalize").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container, 
            "Prepare YOLO", 
            "Split dataset into Train/Val/Test and generate config.", 
            self.run_prepare_yolo
        )

    def setup_model_tab(self):
        """Sets up the Model Training tab."""
        container = tk.Frame(self.tab_model, bg=BG_COLOR, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        self.heading(container, "4. Training").pack(anchor=tk.W, pady=(0, 10))
        
        self.create_action_button(
            container, 
            "Train Model", 
            "Start training YOLO11n-OBB on the prepared dataset (GPU recommended).", 
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
        """Runs a target function in a separate thread to keep UI responsive."""
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
        """Runs the DICOM to PNG conversion process."""
        converter = DicomConverter()
        self.run_in_thread(lambda: converter.run_conversion(progress_callback=self.log), "Converting DICOM files...")

    def run_cleaner(self):
        """Opens the Dataset Cleaner window."""
        self.set_status("Opening Dataset Cleaner...")
        cleaner_window = FabellaCleaner(self.root)
        self.set_status("Ready")

    def run_labeler(self):
        """Runs the OBB Labeler tool."""
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
        
    def run_prepare_yolo(self):
        """Runs the YOLO dataset preparation process."""
        preparer = YoloPreparer()
        self.run_in_thread(lambda: preparer.setup_dataset(progress_callback=self.log), "Preparing YOLO dataset...")

    def run_train_yolo(self):
        """Runs the YOLO model training process."""
        trainer = YoloTrainer()
        self.run_in_thread(lambda: trainer.train_fabella(progress_callback=self.log), "Training YOLO model...")

    def run_test_model(self):
        """Runs the YOLO model testing process."""
        tester = YoloTester()
        self.run_in_thread(lambda: tester.run_test(progress_callback=self.log), "Testing YOLO model...")

if __name__ == "__main__":
    root = tk.Tk()
    app = FabellaApp(root)
    root.mainloop()
