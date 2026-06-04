# Main application module that integrates all tools into a single UI.
import os
import queue
# Suppress verbose warnings from TensorFlow and Albumentations before any imports
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from converter import DicomConverter
from sorter import FabellaCleaner
from labeler_obb import OBBLabeler
from labeler_seg import SegLabeler
from labeler_sam import SAM3AutoLabeler
from preparer import YoloPreparer
from trainer import ModelTrainer, ModelRegistry, ARCHITECTURES
from tester import ModelTester
from prepare_dialog import DatasetGeneratorModal
from shape_analysis import ShapeAnalysisTab
from classifier_utils import CLASSIFIER_ARCH, default_imgsz_for_backbone

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


class AutoLabelerConfigDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Auto-Label Model Setup")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result = None

        self.options = []
        if os.path.exists("rf-detr-medium-seg-trained.pt"):
            self.options.append({
                "label": "RF-DETR Medium Seg (trained)",
                "backend": "rfdetr_seg",
                "path": "rf-detr-medium-seg-trained.pt",
            })
        if os.path.exists("sam3.pt"):
            self.options.append({
                "label": "SAM3 Segment-Everything",
                "backend": "sam3",
                "path": "sam3.pt",
            })

        container = tk.Frame(self, bg=BG_COLOR, padx=18, pady=16)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            container,
            text="Choose which model should propose masks for review.",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=FG_COLOR,
        ).pack(anchor=tk.W, pady=(0, 10))

        tk.Label(container, text="Proposal Model:", bg=BG_COLOR, fg=FG_COLOR).pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(container, textvariable=self.model_var, state="readonly", width=40)
        self.model_cb["values"] = [o["label"] for o in self.options]
        if self.options:
            self.model_cb.current(0)
        self.model_cb.pack(fill=tk.X, pady=(4, 10))
        self.model_cb.bind("<<ComboboxSelected>>", self._sync_threshold_state)

        tk.Label(container, text="Confidence Threshold:", bg=BG_COLOR, fg=FG_COLOR).pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="0.80")
        self.threshold_entry = tk.Entry(
            container,
            textvariable=self.threshold_var,
            width=10,
            bg="#2D2D2D",
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
        )
        self.threshold_entry.pack(anchor=tk.W, pady=(4, 4))

        self.help_label = tk.Label(
            container,
            text=(
                "RF-DETR mode uses the threshold directly. "
                "I set 0.80 as the default because it is a safer starting point "
                "than the older 0.58 recommendation."
            ),
            font=("Segoe UI", 9),
            bg=BG_COLOR,
            fg="#999999",
            justify=tk.LEFT,
            wraplength=420,
        )
        self.help_label.pack(anchor=tk.W, pady=(0, 12))

        # ── Active Compute Limits Frame ──────────────────────────
        compute_frame = tk.LabelFrame(
            container,
            text=" Active Compute Settings (Anti-Freeze) ",
            bg=BG_COLOR,
            fg=ACCENT_COLOR,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=10,
            relief=tk.GROOVE,
        )
        compute_frame.pack(fill=tk.X, pady=(0, 16))

        self.throttle_var = tk.BooleanVar(value=True)
        self.throttle_cb = tk.Checkbutton(
            compute_frame,
            text="Slow down processing while utilizing the PC",
            variable=self.throttle_var,
            bg=BG_COLOR,
            fg=FG_COLOR,
            selectcolor="#2D2D2D",
            activebackground=BG_COLOR,
            activeforeground=FG_COLOR,
            command=self._sync_compute_state,
            font=("Segoe UI", 9)
        )
        self.throttle_cb.pack(anchor=tk.W, pady=(0, 6))

        settings_row = tk.Frame(compute_frame, bg=BG_COLOR)
        settings_row.pack(fill=tk.X)

        tk.Label(settings_row, text="Active Delay (s):", bg=BG_COLOR, fg=FG_COLOR, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="1.5")
        self.delay_entry = tk.Entry(
            settings_row,
            textvariable=self.delay_var,
            width=6,
            bg="#2D2D2D",
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
        )
        self.delay_entry.pack(side=tk.LEFT, padx=(4, 15))

        tk.Label(settings_row, text="Idle Timeout (s):", bg=BG_COLOR, fg=FG_COLOR, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.idle_timeout_var = tk.StringVar(value="5.0")
        self.idle_timeout_entry = tk.Entry(
            settings_row,
            textvariable=self.idle_timeout_var,
            width=6,
            bg="#2D2D2D",
            fg=FG_COLOR,
            insertbackground=FG_COLOR,
        )
        self.idle_timeout_entry.pack(side=tk.LEFT, padx=(4, 0))

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
            text="Start",
            command=self._on_confirm,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground=ACCENT_COLOR,
            relief=tk.FLAT,
            padx=16,
        ).pack(side=tk.RIGHT, padx=(0, 8))

        self._sync_threshold_state()
        self._sync_compute_state()

    def _selected_option(self):
        idx = self.model_cb.current()
        if idx < 0 or idx >= len(self.options):
            return None
        return self.options[idx]

    def _sync_threshold_state(self, _event=None):
        selected = self._selected_option()
        if not selected:
            self.threshold_entry.config(state="disabled")
            return
        if selected["backend"] == "rfdetr_seg":
            self.threshold_entry.config(state="normal")
            self.help_label.config(
                text=(
                    "RF-DETR mode uses the threshold directly. "
                    "I set 0.80 as the default because it is a safer starting point "
                    "than the older 0.58 recommendation."
                )
            )
        else:
            self.threshold_entry.config(state="disabled")
            self.help_label.config(
                text="SAM3 ignores the confidence threshold and runs segment-everything mode."
            )

    def _sync_compute_state(self):
        state = "normal" if self.throttle_var.get() else "disabled"
        self.delay_entry.config(state=state)
        self.idle_timeout_entry.config(state=state)

    def _on_confirm(self):
        selected = self._selected_option()
        if not selected:
            messagebox.showerror("No Model", "No proposal model is available.", parent=self)
            return

        threshold = 0.80
        if selected["backend"] == "rfdetr_seg":
            try:
                threshold = float(self.threshold_var.get())
            except ValueError:
                messagebox.showerror("Invalid Threshold", "Threshold must be numeric.", parent=self)
                return
            if not (0.0 <= threshold <= 1.0):
                messagebox.showerror(
                    "Invalid Threshold",
                    "Threshold must be between 0.0 and 1.0.",
                    parent=self,
                )
                return

        throttle_on_activity = self.throttle_var.get()
        throttle_delay = 1.5
        idle_timeout = 5.0

        if throttle_on_activity:
            try:
                throttle_delay = float(self.delay_var.get())
            except ValueError:
                messagebox.showerror("Invalid Delay", "Active Delay must be numeric (seconds).", parent=self)
                return
            if throttle_delay < 0:
                messagebox.showerror("Invalid Delay", "Active Delay cannot be negative.", parent=self)
                return

            try:
                idle_timeout = float(self.idle_timeout_var.get())
            except ValueError:
                messagebox.showerror("Invalid Timeout", "Idle Timeout must be numeric (seconds).", parent=self)
                return
            if idle_timeout < 0:
                messagebox.showerror("Invalid Timeout", "Idle Timeout cannot be negative.", parent=self)
                return

        self.result = {
            "proposal_backend": selected["backend"],
            "proposal_model_path": selected["path"],
            "proposal_threshold": threshold,
            "throttle_on_activity": throttle_on_activity,
            "throttle_delay": throttle_delay,
            "idle_timeout": idle_timeout,
        }
        self.destroy()


class FabellaApp:
    # The main application class for the Fabella Dataset Manager.
    def __init__(self, root):
        # Initializes the FabellaApp.
        # Args:
        # root (tk.Tk): The root Tkinter window.
        self.root = root
        self.root.title("Fabella Dataset Manager")
        self.root.geometry("1360x920")
        self.root.minsize(1180, 760)
        self.root.configure(bg=BG_COLOR)
        self._ui_task_queue = queue.SimpleQueue()
        self._ui_pump_job = None

        # Folder selection variables with defaults
        self.raw_dir_var = tk.StringVar(value="data/raw")
        self.png_dir_var = tk.StringVar(value="data/png")
        self.sorted_dir_var = tk.StringVar(value="data/sorted")
        self.discarded_dir_var = tk.StringVar(value="data/discarded")
        self.obb_label_dir_var = tk.StringVar(value="data/labels/obb")
        self.seg_label_dir_var = tk.StringVar(value="data/labels/seg")

        # Trace updates to refresh stats immediately when a folder is typed or browsed
        self.raw_dir_var.trace_add("write", lambda *args: self._refresh_dataset_stats())
        self.png_dir_var.trace_add("write", lambda *args: self._refresh_dataset_stats())
        self.sorted_dir_var.trace_add("write", lambda *args: self._refresh_dataset_stats())
        self.discarded_dir_var.trace_add("write", lambda *args: self._refresh_dataset_stats())
        self.obb_label_dir_var.trace_add("write", lambda *args: self._refresh_dataset_stats())
        self.seg_label_dir_var.trace_add("write", lambda *args: self._refresh_dataset_stats())
        
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
        self.tab_shape = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dataset, text="Dataset Tools")
        self.notebook.add(self.tab_model, text="Model Training")
        self.notebook.add(self.tab_shape, text="Shape Analysis")
        
        self.setup_dataset_tab()
        self.setup_model_tab()
        self.setup_shape_analysis_tab()
        
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
        self._schedule_ui_pump()

    def log(self, message):
        # Appends a message to the log text area.
        self._enqueue_ui(lambda message=message: self.update_log_gui(message))

    def update_log_gui(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def set_status(self, message):
        # Safely updates the status bar text from any thread.
        self._enqueue_ui(lambda message=message: self.status_bar.config(text=message))

    def _enqueue_ui(self, callback):
        self._ui_task_queue.put(callback)

    def _schedule_ui_pump(self):
        self._ui_pump_job = self.root.after(50, self._drain_ui_queue)

    def _drain_ui_queue(self):
        self._ui_pump_job = None
        try:
            while True:
                callback = self._ui_task_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        except tk.TclError:
            return

        try:
            if self.root.winfo_exists():
                self._schedule_ui_pump()
        except tk.TclError:
            return

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

    def _browse_dir(self, var, title):
        from tkinter import filedialog
        initial = var.get()
        if not os.path.exists(initial):
            initial = os.getcwd()
        d = filedialog.askdirectory(parent=self.root, title=title, initialdir=initial)
        if d:
            var.set(os.path.normpath(d))
            self._refresh_dataset_stats()

    def setup_dataset_tab(self):
        # Sets up the Dataset Tools tab.
        container = tk.Frame(self.tab_dataset, bg=BG_COLOR, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        # ── Dataset Stats Panel ──────────────────────────────────
        stats_frame = tk.Frame(container, bg="#252526", highlightbackground="#3E3E3E",
                               highlightthickness=1, padx=12, pady=8)
        stats_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(stats_frame, text="Dataset Overview", font=("Segoe UI", 10, "bold"),
                 bg="#252526", fg=ACCENT_COLOR).pack(anchor=tk.W)

        self.stats_label_var = tk.StringVar(
            value="Scanning..."
        )
        tk.Label(stats_frame, textvariable=self.stats_label_var,
                 font=("Consolas", 8), bg="#252526", fg="#BBBBBB",
                 justify=tk.LEFT, anchor=tk.W).pack(fill=tk.X, anchor=tk.W, pady=(4, 0))

        self.root.after(300, self._refresh_dataset_stats)

        # ── Target Folders Configuration Section ─────────────────
        cfg_frame = tk.Frame(container, bg="#252526", highlightbackground="#3E3E3E",
                             highlightthickness=1, padx=12, pady=10)
        cfg_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(cfg_frame, text="Folder Paths Configuration", font=("Segoe UI", 11, "bold"),
                 bg="#252526", fg=ACCENT_COLOR).pack(anchor=tk.W, pady=(0, 10))

        grid_frame = tk.Frame(cfg_frame, bg="#252526")
        grid_frame.pack(fill=tk.X)

        paths = [
            ("Raw DICOM Base", self.raw_dir_var, "Select Raw DICOM Base Directory"),
            ("Output PNG Base", self.png_dir_var, "Select Converted PNG Base Directory"),
            ("Sorted Base", self.sorted_dir_var, "Select Sorted PNG Base Directory"),
            ("Discarded Base", self.discarded_dir_var, "Select Discarded PNG Base Directory"),
            ("OBB Labels Dir", self.obb_label_dir_var, "Select Oriented Bounding Box Labels Directory"),
            ("Seg Labels Dir", self.seg_label_dir_var, "Select Segmentation Labels Directory"),
        ]

        for idx, (label_text, var, title) in enumerate(paths):
            row = idx // 2
            col_offset = 0 if idx % 2 == 0 else 3
            
            tk.Label(grid_frame, text=label_text + ":", bg="#252526", fg=FG_COLOR,
                     font=("Segoe UI", 9)).grid(row=row, column=col_offset, padx=(5, 5), pady=5, sticky=tk.W)
            
            ent = tk.Entry(grid_frame, textvariable=var, width=32, bg="#3C3C3C", fg="white",
                           insertbackground="white", relief=tk.FLAT)
            ent.grid(row=row, column=col_offset+1, padx=(0, 5), pady=5, sticky=tk.EW)
            
            btn = tk.Button(
                grid_frame,
                text="Browse...",
                font=("Segoe UI", 8),
                bg=BUTTON_BG,
                fg="white",
                activebackground=BUTTON_ACTIVE,
                activeforeground="white",
                relief=tk.FLAT,
                command=lambda v=var, t=title: self._browse_dir(v, t),
                padx=8,
                cursor="hand2"
            )
            btn.grid(row=row, column=col_offset+2, padx=(0, 15), pady=5)

        # Make the Entry columns expandable
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.columnconfigure(4, weight=1)

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

        self.create_action_button(
            container,
            "SAM3 Auto-Label",
            "AI-assisted labeling: choose SAM3 or the trained RF-DETR segmentation model, then approve/reject masks.",
            self.run_sam3_auto_labeler,
            color="#6A1B9A"
        )
        
        self.heading(container, "3. Finalize").pack(anchor=tk.W, pady=(20, 10))

        self.create_action_button(
            container,
            "Prepare Dataset",
            "Configure augmentation, then build Train/Val split + YOLO + COCO + Detection formats (all architectures).",
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
                                    values=["n","s","m","l","x"], state="readonly", width=20)
        self.size_cb.grid(row=1, column=1, padx=5, pady=5)
        self.size_cb.bind("<<ComboboxSelected>>", self._on_size_change)

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

        # ── Model picker ─────────────────────────────────────────
        picker_frame = tk.Frame(container, bg=BG_COLOR)
        picker_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(picker_frame, text="Model:", bg=BG_COLOR, fg=FG_COLOR,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))

        self.model_select_var = tk.StringVar(value="")
        self.model_select_cb  = ttk.Combobox(
            picker_frame, textvariable=self.model_select_var,
            state="readonly", width=44
        )
        self.model_select_cb.pack(side=tk.LEFT, padx=(0, 6))
        self.model_select_cb.bind("<<ComboboxSelected>>", self._on_model_select)

        tk.Button(
            picker_frame, text="↻ Refresh",
            bg=BUTTON_BG, fg=FG_COLOR, activebackground=BUTTON_ACTIVE,
            relief=tk.FLAT, padx=8, command=self._refresh_model_list
        ).pack(side=tk.LEFT)

        self.model_info_var = tk.StringVar(value="No models found — train one first.")
        tk.Label(container, textvariable=self.model_info_var,
                 bg=BG_COLOR, fg="#9E9E9E",
                 font=("Segoe UI", 8), anchor=tk.W).pack(fill=tk.X, pady=(0, 8))

        self._available_models = []
        self._refresh_model_list()

        self.create_action_button(
            container,
            "Test Model",
            "Run inference on unsorted images and sort results into detected/undetected.",
            self.run_test_model,
            color="#C62828"
        )

    def setup_shape_analysis_tab(self):
        # Sets up the Shape Analysis tab.
        ShapeAnalysisTab(
            self.tab_shape,
            bg_color=BG_COLOR,
            fg_color=FG_COLOR,
            accent_color=ACCENT_COLOR,
            button_bg=BUTTON_BG,
            button_active=BUTTON_ACTIVE,
            log_callback=self.log,
            status_callback=self.set_status,
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
                self._enqueue_ui(lambda e=e: messagebox.showerror("Error", str(e)))

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    def run_dicom_conversion(self):
        # Runs the DICOM to PNG conversion process with custom folders.
        converter = DicomConverter(
            base_dir=self.raw_dir_var.get(),
            output_base=self.png_dir_var.get()
        )
        self.run_in_thread(lambda: converter.run_conversion(progress_callback=self.log), "Converting DICOM files...")

    def run_cleaner(self):
        # Opens the Dataset Cleaner window for positive images with custom folders.
        self.set_status("Opening Dataset Cleaner...")
        cleaner_window = FabellaCleaner(
            self.root,
            input_base=self.png_dir_var.get(),
            output_base=self.sorted_dir_var.get(),
            discard_base=self.discarded_dir_var.get(),
            category="pos"
        )
        self.set_status("Ready")

    def run_neg_sorter(self):
        # Opens the Dataset Cleaner window scoped to negative images with custom folders.
        # "Keep" = truly negative (no fabella) → Custom Sorted Base/neg
        # "Discard" = actually has a fabella → Custom Sorted Base/pos
        self.set_status("Opening Negative Sorter...")
        FabellaCleaner(
            self.root,
            input_base=self.png_dir_var.get(),
            output_base=self.sorted_dir_var.get(),
            discard_base=self.discarded_dir_var.get(),
            category="neg",
            discard_dir_override=os.path.join(self.sorted_dir_var.get(), "pos"),
            keep_label="NO FABELLA →",
            discard_label="← HAS FABELLA",
            window_title="Negative Sorter — Has Fabella? Move to Pos"
        )
        self.set_status("Ready")

    def run_labeler(self):
        # Runs the OBB Labeler tool with custom folders.
        self._run_cv_tool(
            OBBLabeler(
                image_dir=os.path.join(self.sorted_dir_var.get(), "pos"),
                label_dir=self.obb_label_dir_var.get()
            ),
            "OBB Labeler"
        )

    def run_seg_labeler(self):
        # Runs the Segmentation Labeler tool with custom folders.
        self._run_cv_tool(
            SegLabeler(
                image_dir=os.path.join(self.sorted_dir_var.get(), "pos"),
                label_dir=self.seg_label_dir_var.get()
            ),
            "Segmentation Labeler"
        )

    def run_sam3_auto_labeler(self):
        # Runs the model-assisted auto-labeler with custom folders.
        dialog = AutoLabelerConfigDialog(self.root)
        self.root.wait_window(dialog)
        if not dialog.result:
            return

        params = dialog.result.copy()
        params["image_dir"] = os.path.join(self.sorted_dir_var.get(), "pos")
        params["label_dir"] = self.seg_label_dir_var.get()
        params["parent"] = self.root

        labeler = SAM3AutoLabeler(**params)
        tool_name = "RF-DETR Auto-Labeler" if labeler.proposal_backend == "rfdetr_seg" else "SAM3 Auto-Labeler"
        self._run_cv_tool(labeler, tool_name)
        backend_name = "RF-DETR Seg" if labeler.proposal_backend == "rfdetr_seg" else "SAM3"
        self.log(f"{backend_name} auto-label session: {labeler.stats['approved']} approved, "
                 f"{labeler.stats['rejected']} rejected, "
                 f"{labeler.stats['edited']} manually edited")
        self._refresh_dataset_stats()

    def _run_cv_tool(self, tool, name):
        """Run an OpenCV-based tool in the main thread with error handling."""
        self.set_status(f"Opening {name}...")
        try:
            tool.run()
            self.set_status("Ready")
        except Exception as e:
            self.log(f"Error: {e}")
            self.set_status("Error occurred")
            messagebox.showerror("Error", str(e))

    def _refresh_dataset_stats(self):
        """Scan directories and update the stats panel in the Dataset Tools tab."""
        def _count(d, ext=".png"):
            if not os.path.isdir(d):
                return 0
            try:
                return len([f for f in os.listdir(d) if f.lower().endswith(ext)])
            except Exception:
                return 0

        raw_pos   = _count(os.path.join(self.raw_dir_var.get(), "pos"), ".dcm")
        raw_neg   = _count(os.path.join(self.raw_dir_var.get(), "neg"), ".dcm")
        png_pos   = _count(os.path.join(self.png_dir_var.get(), "pos"))
        png_neg   = _count(os.path.join(self.png_dir_var.get(), "neg"))
        sort_pos  = _count(os.path.join(self.sorted_dir_var.get(), "pos"))
        sort_neg  = _count(os.path.join(self.sorted_dir_var.get(), "neg"))
        lbl_seg   = _count(self.seg_label_dir_var.get(), ".txt")
        lbl_obb   = _count(self.obb_label_dir_var.get(), ".txt")

        # Scan for YOLO Seg/Det configurations and COCO
        try:
            base_parent = os.path.dirname(os.path.dirname(os.path.normpath(self.sorted_dir_var.get())))
            yolo_seg  = os.path.isfile(os.path.join(base_parent, "yolo", "data_seg.yaml")) or os.path.isfile("data/yolo/data_seg.yaml")
            yolo_det  = os.path.isfile(os.path.join(base_parent, "yolo", "data_det.yaml")) or os.path.isfile("data/yolo/data_det.yaml")
            coco_ok   = os.path.isdir(os.path.join(base_parent, "coco", "train")) or os.path.isdir("data/coco/train")
        except Exception:
            yolo_seg  = os.path.isfile("data/yolo/data_seg.yaml")
            yolo_det  = os.path.isfile("data/yolo/data_det.yaml")
            coco_ok   = os.path.isdir("data/coco/train")

        parts = []
        parts.append(f"Raw DICOM:  {raw_pos} pos  |  {raw_neg} neg")
        parts.append(f"PNG:        {png_pos} pos  |  {png_neg} neg")
        parts.append(f"Sorted:     {sort_pos} pos  |  {sort_neg} neg")
        parts.append(f"Labels:     {lbl_seg} seg  |  {lbl_obb} obb")

        fmt_parts = []
        if yolo_seg: fmt_parts.append("YOLO Seg")
        if yolo_det: fmt_parts.append("YOLO Det")
        if coco_ok:  fmt_parts.append("COCO")
        fmt_str = ", ".join(fmt_parts) if fmt_parts else "Not prepared"
        parts.append(f"Prepared:   {fmt_str}")

        try:
            self.stats_label_var.set("\n".join(parts))
        except Exception:
            pass
        
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
        self._on_size_change()

    def _on_size_change(self, _event=None):
        arch = self.arch_var.get()
        if arch == CLASSIFIER_ARCH:
            backbone = self.size_var.get()
            try:
                self.imgsz_var.set(str(default_imgsz_for_backbone(backbone)))
            except Exception:
                self.imgsz_var.set("384")

    def _get_arch_info(self):
        # Returns (arch, version, size, epochs, imgsz, batch) from current UI state.
        return (
            self.arch_var.get(),
            self.version_var.get(),
            self.size_var.get(),
            int(self.epochs_var.get() or 100),
            int(self.imgsz_var.get() or 1024),
            int(self.batch_var.get() or 4),
        )

    def _get_yolo_task(self):
        # Maps current architecture to task key used by YoloPreparer.
        arch = self.arch_var.get()
        if arch == CLASSIFIER_ARCH:
            return "classify"
        if arch == "YOLO OBB":
            return "obb"
        if arch == "RT-DETR":
            return "detect"
        return "segment"  # YOLO Seg default

    def run_prepare_dataset(self):
        # Opens the augmentation modal, then shows a progress window while preparing.
        arch = self.arch_var.get()
        task = self._get_yolo_task()
        if task == "classify":
            messagebox.showinfo(
                "Classifier Training",
                "Torchvision Classifier trains directly from data/sorted/pos and "
                "data/sorted/neg, so the YOLO/COCO prepare step is not needed.",
                parent=self.root,
            )
            return

        def on_generate(config):
            prog = PrepareProgressWindow(self.root, title=f"Preparing Dataset — All Formats")

            def step_cb(pct, label):
                self._enqueue_ui(lambda pct=pct, label=label: prog.update_step(pct, label))

            def log_cb(msg):
                self._enqueue_ui(lambda msg=msg: self.update_log_gui(msg))
                self._enqueue_ui(lambda msg=msg: prog.append_log(msg))

            def run():
                self.set_status(f"Preparing dataset (YOLO + COCO + Detection)...")
                try:
                    custom_label_dir = (
                        self.obb_label_dir_var.get()
                        if task == "obb"
                        else self.seg_label_dir_var.get()
                    )
                    preparer = YoloPreparer(
                        task=task,
                        pos_img_dir=os.path.join(self.sorted_dir_var.get(), "pos"),
                        neg_dicom_dir=os.path.join(self.raw_dir_var.get(), "neg"),
                        label_dir=custom_label_dir
                    )
                    preparer.setup_dataset(
                        config=config,
                        progress_callback=log_cb,
                        step_callback=step_cb,
                    )
                    self.set_status("Ready")
                    # Refresh stats panel
                    self._enqueue_ui(self._refresh_dataset_stats)
                except Exception as e:
                    self.log(f"Error: {e}")
                    self.set_status("Error occurred")
                    self._enqueue_ui(lambda e=e: messagebox.showerror("Error", str(e)))

            t = threading.Thread(target=run, daemon=True)
            t.start()

        DatasetGeneratorModal(
            self.root,
            task=task,
            pos_sorted_dir=os.path.join(self.sorted_dir_var.get(), "pos"),
            on_generate=on_generate
        )

    def run_train_model(self):
        # Trains the selected architecture.  RF-DETR logs to TensorBoard
        # automatically; YOLO logs to its own results.csv / plots.
        arch, version, size, epochs, imgsz, batch = self._get_arch_info()
        trainer = ModelTrainer(
            arch=arch, version=version, size=size,
            epochs=epochs, imgsz=imgsz, batch=batch,
            pos_dir=os.path.join(self.sorted_dir_var.get(), "pos"),
            neg_dir=os.path.join(self.sorted_dir_var.get(), "neg")
        )

        os.makedirs(trainer.run_dir, exist_ok=True)

        def do_train():
            trainer.train(progress_callback=self.log)
            path = getattr(trainer, 'results_plot_path', None)
            if path and os.path.exists(path):
                self._enqueue_ui(lambda path=path: self._show_results_plot(path))
            # Refresh model selector now that a new run is registered
            self._enqueue_ui(self._refresh_model_list)

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

    def _refresh_model_list(self):
        """Scan registry + disk and populate the model selector combobox."""
        models = ModelRegistry.all_models()
        self._available_models = models
        if not models:
            self.model_select_cb.config(values=[])
            self.model_select_var.set("")
            self.model_info_var.set("No models found — train one first.")
            return

        labels = []
        for m in models:
            version = f" v{m['version']}" if m.get("version") else ""
            labels.append(f"{m['arch']}{version} [{m['size']}] — {m['date']}")

        self.model_select_cb.config(values=labels)
        self.model_select_var.set(labels[0])
        self._on_model_select()

    def _on_model_select(self, _event=None):
        """Update the info label when a model is chosen from the dropdown."""
        idx = self.model_select_cb.current()
        if idx < 0 or idx >= len(self._available_models):
            return
        m = self._available_models[idx]
        exists = "found" if os.path.exists(m.get("weights_path", "")) else "missing"
        self.model_info_var.set(
            f"Run: {m['run_name']}  |  Epochs: {m.get('epochs','?')}  |  "
            f"ImgSize: {m.get('imgsz','?')}  |  Batch: {m.get('batch','?')}  |  weights {exists}"
        )

    def run_test_model(self):
        idx = self.model_select_cb.current()
        if not self._available_models:
            messagebox.showwarning("No Model", "No trained models found. Train a model first.")
            return
        if idx < 0:
            idx = 0
        entry = self._available_models[idx]
        arch  = entry.get("arch", "YOLO Seg")
        size  = entry.get("size", "n")
        weights = entry.get("weights_path", "")
        
        # Instantiate ModelTester with customized paths representing custom PNG and custom Sorted directory
        tester = ModelTester(
            arch=arch,
            size=size,
            src_dir=os.path.join(self.png_dir_var.get(), "pos"),
            sorted_dir=os.path.join(self.sorted_dir_var.get(), "pos")
        )
        if weights and os.path.exists(weights):
            tester.model_path = weights
        self.run_in_thread(
            lambda: tester.run_test(progress_callback=self.log),
            f"Testing {arch} [{size}]..."
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = FabellaApp(root)
    root.mainloop()
