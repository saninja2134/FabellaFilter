# Unified model trainer supporting YOLO (seg/obb), RT-DETR, and RF-DETR architectures.
import os
import sys
import json
import torch
from datetime import datetime

REGISTRY_PATH = os.path.join("output", "model_registry.json")


# ─────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Persistent JSON registry of every trained model run.
    Also scans output/runs/ for unregistered models found on disk."""

    @staticmethod
    def load():
        if os.path.exists(REGISTRY_PATH):
            try:
                with open(REGISTRY_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    @staticmethod
    def register(entry):
        """Insert or replace entry by run_name (newest-first order)."""
        records = ModelRegistry.load()
        records = [r for r in records if r.get("run_name") != entry["run_name"]]
        records.insert(0, entry)
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(records, f, indent=2)

    @staticmethod
    def all_models():
        """Return registry entries + any unregistered on-disk runs."""
        records = ModelRegistry.load()
        known = {r["run_name"] for r in records}
        runs_dir = os.path.join("output", "runs")
        if os.path.isdir(runs_dir):
            for name in sorted(os.listdir(runs_dir)):
                if name in known:
                    continue
                run_path = os.path.join(runs_dir, name)
                # Ultralytics: weights/best.pt
                weights = os.path.join(run_path, "weights", "best.pt")
                if not os.path.exists(weights):
                    # RF-DETR: checkpoint_best_ema.pth at run root
                    for ckpt in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                                 "checkpoint_best_total.pth", "checkpoint.pth"):
                        candidate = os.path.join(run_path, ckpt)
                        if os.path.exists(candidate):
                            weights = candidate
                            break
                if not os.path.exists(weights):
                    continue
                arch, size = ModelRegistry._infer_from_name(name)
                records.append({
                    "run_name":     name,
                    "arch":         arch,
                    "version":      "",
                    "size":         size,
                    "epochs":       "?",
                    "imgsz":        "?",
                    "batch":        "?",
                    "date":         "unknown",
                    "weights_path": weights,
                    "results_plot": os.path.join(run_path, "results.png"),
                })
        return records

    @staticmethod
    def _infer_from_name(name):
        """Best-effort parse of run_name like fabella_yolo_seg_n_v1."""
        stripped = name
        if stripped.startswith("fabella_"):
            stripped = stripped[len("fabella_"):]
        if "_v" in stripped:
            stripped = stripped[:stripped.rfind("_v")]
        parts = stripped.rsplit("_", 1)
        size     = parts[-1] if len(parts) > 1 else "?"
        arch_key = parts[0]  if len(parts) > 1 else stripped
        arch_map = {
            "yolo_seg":    "YOLO Seg",
            "yolo_obb":    "YOLO OBB",
            "rt_detr":     "RT-DETR",
            "rf_detr":     "RF-DETR",
            "rf_detr_seg": "RF-DETR Seg",
        }
        return arch_map.get(arch_key, arch_key), size


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
        "sizes":    ["n", "s", "m", "l"],
        "versions": [""],
    },
    "RF-DETR Seg": {
        "backend":  "rfdetr",
        "task":     "segment",
        "format":   "coco",
        "sizes":    ["n", "s", "m", "l", "xl", "2xl"],
        "versions": [""],
    },
}


def get_arch_info(arch):
    # Returns architecture metadata dict or raises if unknown.
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch]


# RF-DETR class name maps (shared by trainer + tester)
RFDETR_DET_MAP = {
    "n": "RFDETRNano",   "s": "RFDETRSmall",
    "m": "RFDETRMedium", "l": "RFDETRLarge",
}
RFDETR_SEG_MAP = {
    "n": "RFDETRSegNano",   "s": "RFDETRSegSmall",
    "m": "RFDETRSegMedium", "l": "RFDETRSegLarge",
    "xl": "RFDETRSegXLarge", "2xl": "RFDETRSeg2XLarge",
}


def get_rfdetr_class(task, size):
    """Return (class_name, class_object) for an RF-DETR variant, or raise."""
    cls_map = RFDETR_SEG_MAP if task == "segment" else RFDETR_DET_MAP
    cls_name = cls_map.get(size)
    if cls_name is None:
        raise ValueError(f"Unknown RF-DETR size '{size}'. Options: {list(cls_map)}")
    import rfdetr as _rfdetr_mod
    return cls_name, getattr(_rfdetr_mod, cls_name)


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
        base_name = f"fabella_{safe_arch}_{size}"

        # Auto-version: scan existing runs and pick next version number
        self.output_dir = "output/runs"
        next_v = 1
        if os.path.isdir(self.output_dir):
            for name in os.listdir(self.output_dir):
                if name.startswith(base_name + "_v"):
                    try:
                        v = int(name.split("_v")[-1])
                        next_v = max(next_v, v + 1)
                    except ValueError:
                        pass
        self.run_name   = f"{base_name}_v{next_v}"
        self.data_yaml  = f"data/yolo/data_{task_key}.yaml"
        self.coco_dir   = f"data/coco"
        self.results_plot_path = None   # set after training if a chart was generated

    @property
    def run_dir(self):
        """Full path to this training run's output directory."""
        return os.path.join(self.output_dir, self.run_name)
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
            patience=30,  # Small objects need longer to converge
            save=True,

            # ── Augmentation: tuned for small sesamoid bone on X-ray ──
            mosaic=0.0,       # OFF — mosaic shrinks already-tiny fabella
            mixup=0.0,        # OFF — blurs subtle bone density contrast
            copy_paste=0.0,   # OFF — anatomically invalid random placement
            erasing=0.0,      # OFF — would occlude the tiny target

            # Geometric augmentation (mild — anatomy is position-sensitive)
            scale=0.4,        # Scale variation for different knee sizes
            shear=5,          # Mild shear
            degrees=5.0,      # Slight rotation for patient positioning variation
            translate=0.15,   # Positional shift — fabella location varies

            # Flip policy for lateral knee X-rays
            fliplr=0.5,       # OK — handles left/right knee films
            flipud=0.0,       # OFF — vertical flip breaks anatomical orientation

            # X-rays are grayscale — disable colour augmentation
            hsv_h=0.0,        # No hue (meaningless on grayscale)
            hsv_s=0.0,        # No saturation (meaningless on grayscale)
            hsv_v=0.15,       # Mild brightness — simulates exposure variation

            # Loss weights: emphasise localisation of <1% image-area target
            cls=1.5,          # Classification loss weight
            box=10.0,         # Bbox loss — high for precise small-object localisation
            seg=2.5,          # Seg loss — upweighted because mask area is tiny
            dfl=1.5,          # Distribution focal loss
        )

        best = os.path.join(self.output_dir, self.run_name, "weights", "best.pt")
        plot = os.path.join(self.output_dir, self.run_name, "results.png")
        if os.path.exists(plot):
            self.results_plot_path = plot
        ModelRegistry.register(self._make_registry_entry())
        log(f"\nTraining complete! Best model: {best}")

    def _train_rfdetr(self, log):
        # Patch supervision < 0.26 missing xyxy_to_xywh (rfdetr still needs it)
        try:
            import supervision as _sv
            if not hasattr(_sv, 'xyxy_to_xywh'):
                import numpy as _np
                _sv.xyxy_to_xywh = lambda xyxy: _np.column_stack([
                    xyxy[..., :2], xyxy[..., 2:] - xyxy[..., :2]])
        except ImportError:
            pass

        # ── Dynamic class selection ──
        try:
            cls_name, model_cls = get_rfdetr_class(self.task, self.size)
        except ValueError as e:
            log(f"Error: {e}")
            return
        except (ImportError, ModuleNotFoundError) as e:
            log(f"Error: Cannot import rfdetr ({e})")
            log("  pip install -U rfdetr")
            return
        except AttributeError:
            log(f"Error: RF-DETR class not found. For XL/2XL: pip install rfdetr[plus]")
            return

        if not os.path.isdir(os.path.join(self.coco_dir, "train")):
            # Try auto-generating from existing YOLO data
            yolo_seg = os.path.join("data", "yolo", "seg")
            if os.path.isdir(yolo_seg):
                log("COCO format not found — auto-generating from YOLO data...")
                try:
                    from preparer import YoloPreparer
                    YoloPreparer(task="segment").export_coco(log)
                except Exception as e:
                    log(f"Error generating COCO: {e}")
                    return
            else:
                log(f"Error: {self.coco_dir} not found. Run 'Prepare Dataset' first.")
                return

        log(f"Loading RF-DETR {cls_name}...")

        # Snap resolution to nearest multiple of 32 (patch_size × num_windows)
        divisor = 32
        resolution = max(divisor, round(self.imgsz / divisor) * divisor)

        # Use default constructor (loads correct pretrained weights automatically)
        model = model_cls()

        # Effective batch = batch_size × grad_accum_steps
        grad_accum = max(1, 16 // self.batch)
        warmup = min(3, max(1, self.epochs // 10))

        log(f"  Resolution: {resolution}  |  Effective batch: {self.batch}×{grad_accum}={self.batch * grad_accum}")
        log(f"  Warmup: {warmup} epochs  |  Early stopping patience: 20")
        log("  TensorBoard logging enabled — run: tensorboard --logdir output/runs")

        try:
            model.train(
                # ── Dataset ──
                dataset_dir=self.coco_dir,

                # ── Core training ──
                epochs=self.epochs,
                batch_size=self.batch,
                grad_accum_steps=grad_accum,
                lr=1e-4,
                warmup_epochs=warmup,

                # ── EMA ──
                use_ema=True,
                ema_decay=0.9997,
                ema_tau=100,

                # ── Early stopping ──
                early_stopping=True,
                early_stopping_patience=20,
                early_stopping_min_delta=0.001,
                early_stopping_use_ema=True,

                # ── Checkpoints / output ──
                output_dir=os.path.join(self.output_dir, self.run_name),
                checkpoint_interval=10,

                # ── Resolution ──
                resolution=resolution,

                # ── Logging ──
                tensorboard=True,

                # ── Run test set after training ──
                run_test=True,
            )
            log(f"\nRF-DETR training complete! Output: {self.output_dir}/{self.run_name}")
            ModelRegistry.register(self._make_registry_entry())
        except Exception as e:
            log(f"Training error: {e}")

    def _make_registry_entry(self):
        run_dir = os.path.join(self.output_dir, self.run_name)
        # Ultralytics saves to weights/best.pt; RF-DETR saves checkpoint_best_ema.pth at run root
        if self.backend == "rfdetr":
            weights = self._find_rfdetr_best(run_dir)
        else:
            weights = os.path.join(run_dir, "weights", "best.pt")
        return {
            "run_name":     self.run_name,
            "arch":         self.arch,
            "version":      self.version,
            "size":         self.size,
            "epochs":       self.epochs,
            "imgsz":        self.imgsz,
            "batch":        self.batch,
            "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "weights_path": weights.replace(os.sep, "/"),
            "results_plot": os.path.join(run_dir, "results.png").replace(os.sep, "/"),
        }

    @staticmethod
    def _find_rfdetr_best(run_dir):
        """Return the best RF-DETR checkpoint path (prefer EMA > regular > total > last)."""
        for name in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth",
                     "checkpoint_best_total.pth", "checkpoint.pth"):
            p = os.path.join(run_dir, name)
            if os.path.exists(p):
                return p
        return os.path.join(run_dir, "checkpoint_best_ema.pth")  # fallback