# Unified model trainer supporting YOLO (seg/obb), RT-DETR, and RF-DETR architectures.
import os
import re
import sys
import torch

# ─────────────────────────────────────────────────────────────────
# Registry: defines every supported architecture
# ─────────────────────────────────────────────────────────────────

# arch_key → {backend, task, format, size_options, version_options}
ARCHITECTURES = {
    "YOLO Seg": {
        "backend":  "ultralytics",
        "task":     "segment",
        "format":   "yolo",
        "sizes":    ["n", "s", "m", "l", "x"],
        "versions": ["8", "9", "10", "11", "12", "26"],
    },
    "YOLO OBB": {
        "backend":  "ultralytics",
        "task":     "obb",
        "format":   "yolo",
        "sizes":    ["n", "s", "m", "l", "x"],
        "versions": ["8", "11"],
    },
    "RT-DETR": {
        "backend":  "ultralytics",
        "task":     "detect",
        "format":   "yolo",
        "sizes":    ["l", "x"],
        "versions": [""],   # no numeric version
    },
    "RF-DETR": {
        "backend":  "rfdetr",
        "task":     "detect",
        "format":   "coco",
        "sizes":    ["base", "large"],
        "versions": [""],
    },
}


def get_arch_info(arch):
    # Returns architecture metadata dict or raises if unknown.
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch]


def build_yolo_model_name(arch, version, size):
    # Constructs the ultralytics weight filename.
    if arch == "RT-DETR":
        return f"rtdetr-{size}.pt"
    task_suffix = "-obb" if arch == "YOLO OBB" else "-seg"
    prefix = "yolov" if version == "8" else "yolo"
    return f"{prefix}{version}{size}{task_suffix}.pt"


class ModelTrainer:
    # Unified trainer. Architecture-aware dispatch at runtime.
    def __init__(self, arch="YOLO Seg", version="11", size="n",
                 epochs=100, imgsz=1024, batch=4):
        # Args:
        # arch    : one of ARCHITECTURES keys
        # version : model version string (ignored for RT-DETR/RF-DETR)
        # size    : size token — 'n','s','m','l','x' for YOLO; 'base'/'large' for RF-DETR
        # epochs  : training epochs
        # imgsz   : input image size (square)
        # batch   : batch size
        info = get_arch_info(arch)
        self.arch    = arch
        self.version = version
        self.size    = size
        self.backend = info["backend"]
        self.task    = info["task"]
        self.format  = info["format"]
        self.epochs  = epochs
        self.imgsz   = imgsz
        self.batch   = batch

        # Derive paths
        task_key = "obb" if self.task == "obb" else ("seg" if self.task == "segment" else "det")
        safe_arch = arch.lower().replace(" ", "_").replace("-", "_")
        self.run_name   = f"fabella_{safe_arch}_{size}_v1"
        self.output_dir = "output/runs"
        self.data_yaml  = f"data/yolo/data_{task_key}.yaml"
        self.coco_dir   = f"data/coco"
        self.results_plot_path = None   # set after training if a chart was generated

    # ── PUBLIC API ────────────────────────────────────────────────

    def train(self, progress_callback=None):
        # Dispatch to the correct backend.
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

        self._log_hardware(log)

        if self.backend == "ultralytics":
            self._train_ultralytics(log)
        elif self.backend == "rfdetr":
            self._train_rfdetr(log)
        else:
            log(f"Unknown backend: {self.backend}")

    # ── BACKENDS ─────────────────────────────────────────────────

    def _log_hardware(self, log):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log("--- Hardware Status ---")
        log(f"Device: {device}")
        if device == "cuda":
            log(f"GPU: {torch.cuda.get_device_name(0)}")
            log(f"CUDA Version: {torch.version.cuda}")
        log("-----------------------\n")

    def _train_ultralytics(self, log):
        from ultralytics import YOLO

        model_name = build_yolo_model_name(self.arch, self.version, self.size)

        if not os.path.exists(self.data_yaml):
            log(f"Error: {self.data_yaml} not found. Run 'Prepare Dataset' first.")
            return

        log(f"Loading {model_name}...")
        try:
            model = YOLO(model_name)
        except Exception as e:
            fallback = "yolov8n-seg.pt" if self.task == "segment" else \
                       "yolov8n-obb.pt"  if self.task == "obb"     else \
                       "rtdetr-l.pt"
            log(f"Could not load {model_name}: {e}")
            log(f"Falling back to {fallback}...")
            model = YOLO(fallback)

        device = 0 if torch.cuda.is_available() else "cpu"
        log(f"Starting {self.arch} training (epochs={self.epochs}, imgsz={self.imgsz}, batch={self.batch})...")

        model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=device,
            workers=4,
            project=self.output_dir,
            name=self.run_name,
            patience=20,
            save=True,
            fliplr=0.5,
            flipud=0.2,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            mixup=0.1,
            mosaic=1.0,
        )

        best = os.path.join(self.output_dir, self.run_name, "weights", "best.pt")
        plot = os.path.join(self.output_dir, self.run_name, "results.png")
        if os.path.exists(plot):
            self.results_plot_path = plot
        log(f"\nTraining complete! Best model: {best}")

    def _train_rfdetr(self, log):
        # Patch supervision < 0.26 which dropped xyxy_to_xywh (rfdetr still needs it)
        try:
            import supervision as _sv
            import numpy as _np
            if not hasattr(_sv, 'xyxy_to_xywh'):
                def _xyxy_to_xywh(xyxy):
                    arr = _np.array(xyxy, dtype=float)
                    result = arr.copy()
                    result[..., 2] = arr[..., 2] - arr[..., 0]  # w = x2 - x1
                    result[..., 3] = arr[..., 3] - arr[..., 1]  # h = y2 - y1
                    return result
                _sv.xyxy_to_xywh = _xyxy_to_xywh
        except ImportError:
            pass

        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except (ImportError, ModuleNotFoundError) as e:
            log(f"Error: Cannot import rfdetr ({e})")
            log("This is usually a dependency conflict. Try:")
            log("  pip install -U transformers peft")
            log("  pip install -U rfdetr")
            return

        if not os.path.exists(self.coco_dir):
            log(f"Error: {self.coco_dir} not found. Run 'Prepare Dataset' first (RF-DETR needs COCO format).")
            return

        # RF-DETR expects _annotations.coco.json inside each split folder.
        # Roboflow layout: train/ and valid/ (not val/).
        import shutil as _shutil
        # Rename val/ -> valid/ if it exists and valid/ doesn't yet
        old_val = os.path.join(self.coco_dir, "val")
        new_val = os.path.join(self.coco_dir, "valid")
        if os.path.isdir(old_val) and not os.path.isdir(new_val):
            os.rename(old_val, new_val)
            log("  Renamed data/coco/val -> data/coco/valid")

        # Auto-migrate from the standard annotations/ layout if _annotations.coco.json is missing
        split_map = {"train": "train", "val": "valid"}
        for split, split_dst in split_map.items():
            rfdetr_ann = os.path.join(self.coco_dir, split_dst, "_annotations.coco.json")
            fallback    = os.path.join(self.coco_dir, "annotations", f"instances_{split}.json")
            if not os.path.exists(rfdetr_ann):
                os.makedirs(os.path.join(self.coco_dir, split_dst), exist_ok=True)
                if os.path.exists(fallback):
                    _shutil.copy(fallback, rfdetr_ann)
                    log(f"  Migrated {split} annotations to {split_dst}/_annotations.coco.json")
                else:
                    log(f"Warning: no annotations found for '{split}' split — re-run Prepare Dataset.")

        # RF-DETR also requires test/ split — mirror valid/ if test/ is absent
        valid_dir = os.path.join(self.coco_dir, "valid")
        test_dir  = os.path.join(self.coco_dir, "test")
        if os.path.isdir(valid_dir) and not os.path.isdir(test_dir):
            _shutil.copytree(valid_dir, test_dir)
            log("  Created data/coco/test/ (mirrored from valid/)")


        log(f"Loading RF-DETR {self.size}...")

        # Suppress noisy warnings that spam from dataloader worker processes
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
        os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

        model = RFDETRLarge() if self.size == "large" else RFDETRBase()

        log(f"Starting RF-DETR training (epochs={self.epochs}, batch={self.batch})...")

        # ── Capture stdout+stderr to parse per-epoch metrics ──────
        class _TeeStream:
            def __init__(self, orig, lines):
                self._orig  = orig
                self._lines = lines
                self._buf   = ""
            def write(self, data):
                self._orig.write(data)
                self._buf += data
                while '\n' in self._buf:
                    line, self._buf = self._buf.split('\n', 1)
                    self._lines.append(line)
                return len(data)
            def flush(self): self._orig.flush()
            def __getattr__(self, n): return getattr(self._orig, n)

        captured = []
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_out, captured)
        sys.stderr = _TeeStream(old_err, captured)
        try:
            model.train(
                dataset_dir=self.coco_dir,
                epochs=self.epochs,
                batch_size=self.batch,
                grad_accum_steps=max(1, 8 // self.batch),
                lr=1e-4,
                output_dir=os.path.join(self.output_dir, self.run_name),
                run_test=False,
            )
            log(f"\nRF-DETR training complete! Output: {self.output_dir}/{self.run_name}")
        except Exception as e:
            log(f"Training error: {e}")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        # Parse epoch summary lines and generate results chart
        epoch_metrics = []
        for line in captured:
            m = re.search(
                r'Epoch (\d+) stats:.*\bclass_error:\s*([\d.]+).*\bloss:\s*[\d.]+\s*\(([\d.]+)\)',
                line
            )
            if m:
                epoch_metrics.append({
                    'epoch': int(m.group(1)),
                    'class_error': float(m.group(2)),
                    'loss': float(m.group(3)),
                })
        if epoch_metrics:
            run_dir = os.path.join(self.output_dir, self.run_name)
            plot = self._plot_rfdetr_results(epoch_metrics, run_dir)
            if plot:
                self.results_plot_path = plot
                log(f"Results chart saved: {plot}")
    def _plot_rfdetr_results(self, metrics, out_dir):
        """Generate and save a training results chart from per-epoch metrics."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            epochs = [m['epoch'] for m in metrics]
            losses = [m['loss'] for m in metrics]
            errors = [m['class_error'] for m in metrics]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'RF-DETR Training — {self.run_name}', fontsize=13, fontweight='bold')

            axes[0].plot(epochs, losses, color='#2196F3', linewidth=1.8)
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs, errors, color='#F44336', linewidth=1.8)
            axes[1].set_title('Class Error (%)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Error %')
            axes[1].set_ylim(0, 105)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, 'results.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return path
        except Exception as e:
            print(f"[plot] {e}")
            return None