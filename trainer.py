# Unified model trainer supporting YOLO (seg/obb), RT-DETR, RF-DETR, and classification architectures.
import os
import sys
import json
import torch
from datetime import datetime

from classifier_utils import (
    CLASSIFIER_ARCH,
    DEFAULT_AUTO_POSITIVE_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    format_backbone_label,
    gather_sorted_samples,
    create_classifier_model,
    build_classifier_transforms,
    split_classifier_samples,
    ImagePathDataset,
    save_classifier_checkpoint,
)

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
                    candidate = os.path.join(run_path, "best_classifier.pth")
                    if os.path.exists(candidate):
                        weights = candidate
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
        if stripped.startswith("torchvision_classifier_"):
            arch_key = "torchvision_classifier"
            size = stripped[len("torchvision_classifier_"):]
            return CLASSIFIER_ARCH, size
        parts = stripped.rsplit("_", 1)
        size     = parts[-1] if len(parts) > 1 else "?"
        arch_key = parts[0]  if len(parts) > 1 else stripped
        arch_map = {
            "yolo_seg":    "YOLO Seg",
            "yolo_obb":    "YOLO OBB",
            "rt_detr":     "RT-DETR",
            "rf_detr":     "RF-DETR",
            "rf_detr_seg": "RF-DETR Seg",
            "torchvision_classifier": CLASSIFIER_ARCH,
            "nn_u_net":    "nnU-Net",
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
    CLASSIFIER_ARCH: {
        "backend":  "torchvision",
        "task":     "classify",
        "format":   "sorted_dirs",
        "sizes":    [
            "efficientnet_v2_s",
            "resnet50",
            "resnet18",
            "mobilenet_v3_small",
        ],
        "versions": [""],
    },
    "nnU-Net": {
        "backend":  "nnunet",
        "task":     "segment",
        "format":   "nnunet",
        "sizes":    ["2d", "3d_fullres", "3d_lowres"],
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

RFDETR_SEG_TRAIN_KWARGS = {
    "lr": 1e-4,
    "lr_scheduler": "step",
    "warmup_epochs": 3.0,
    "use_ema": True,
    "ema_decay": 0.9997,
    "ema_tau": 100,
    "early_stopping": True,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.001,
    "early_stopping_use_ema": True,
    "checkpoint_interval": 10,
    "tensorboard": True,
    "wandb": False,
    "run_test": True,
    "num_workers": 2,
    "seed": 42,
}

RFDETR_SEG_LOSS_KWARGS = {
    "mask_ce_loss_coef": 5.0,
    "mask_dice_loss_coef": 5.0,
    "cls_loss_coef": 5.0,
}

# nnU-Net dataset identity (shared by trainer + tester)
NNUNET_DATASET_ID   = 1
NNUNET_DATASET_NAME = f"Dataset{NNUNET_DATASET_ID:03d}_Fabella"


def get_rfdetr_class(task, size):
    # Return (class_name, class_object) for an RF-DETR variant, or raise.
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
                 epochs=100, imgsz=1024, batch=4,
                 pos_dir="data/sorted/pos", neg_dir="data/sorted/neg"):
        # Args:
        # arch    : one of ARCHITECTURES keys
        # version : model version string (ignored for RT-DETR/RF-DETR)
        # size    : size token — 'n','s','m','l','x' for YOLO;
        #           'n','s','m','l' for RF-DETR detect;
        #           'n','s','m','l','xl','2xl' for RF-DETR Seg
        # epochs  : training epochs
        # imgsz   : input image size (square)
        # batch   : batch size
        # pos_dir : directory of sorted positive images
        # neg_dir : directory of sorted negative images
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
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir

        # Derive paths
        if self.task == "obb":
            task_key = "obb"
        elif self.task == "segment":
            task_key = "seg"
        elif self.task == "classify":
            task_key = "cls"
        else:
            task_key = "det"
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
        
        # Derive paths relative to customized directories
        try:
            base_parent = os.path.dirname(os.path.dirname(os.path.normpath(pos_dir)))
            self.data_yaml  = os.path.join(base_parent, "yolo", f"data_{task_key}.yaml")
            self.coco_dir   = os.path.join(base_parent, "coco")
        except Exception:
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
        elif self.backend == "torchvision":
            self._train_torchvision_classifier(log)
        elif self.backend == "nnunet":
            self._train_nnunet(log)
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
        # Patch supervision < 0.26 missing xyxy_to_xywh (rfdetr still needs it).
        try:
            import supervision as _sv
            if not hasattr(_sv, 'xyxy_to_xywh'):
                def _xyxy_to_xywh(xyxy):
                    xywh = xyxy.copy()
                    xywh[..., 2] = xyxy[..., 2] - xyxy[..., 0]
                    xywh[..., 3] = xyxy[..., 3] - xyxy[..., 1]
                    return xywh

                _sv.xyxy_to_xywh = _xyxy_to_xywh
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

        # Use the model's native pretrained settings instead of rounding the UI size.
        # This keeps RF-DETR Seg Small on its default 384x384 resolution.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_cls()
        resolution = getattr(model.model_config, "resolution", self.imgsz)
        pretrain_weights = getattr(model.model_config, "pretrain_weights", None)

        if pretrain_weights:
            log(f"  Pretrained weights: {pretrain_weights}")
        log(f"  Native resolution: {resolution} (requested imgsz={self.imgsz})")

        # Effective batch = batch_size × grad_accum_steps
        grad_accum = max(1, 16 // self.batch)
        warmup = RFDETR_SEG_TRAIN_KWARGS["warmup_epochs"]

        log(f"  Effective batch: {self.batch}×{grad_accum}={self.batch * grad_accum}")
        log(f"  Warmup: {warmup} epochs  |  Early stopping patience: 20")
        log(f"  COCO dataset: {self.coco_dir}")
        log("  TensorBoard logging enabled — run: tensorboard --logdir output/runs")

        try:
            train_kwargs = dict(
                # ── Dataset ──
                dataset_dir=self.coco_dir,

                # ── Core training ──
                epochs=self.epochs,
                batch_size=self.batch,
                grad_accum_steps=grad_accum,
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
                wandb=False,
                device=device,
                num_workers=2,
                seed=42,

                # ── Run test set after training ──
                run_test=True,
            )

            train_kwargs.update(RFDETR_SEG_TRAIN_KWARGS)
            if self.task == "segment":
                train_kwargs.update(RFDETR_SEG_LOSS_KWARGS)

            model.train(**train_kwargs)
            log(f"\nRF-DETR training complete! Output: {self.output_dir}/{self.run_name}")
            ModelRegistry.register(self._make_registry_entry())
        except Exception as e:
            log(f"Training error: {e}")

    def _train_torchvision_classifier(self, log):
        import copy
        import math

        import matplotlib.pyplot as plt
        from torch.utils.data import DataLoader

        samples = gather_sorted_samples()
        pos_count = sum(1 for _, label in samples if label == 1)
        neg_count = sum(1 for _, label in samples if label == 0)
        if pos_count < 2 or neg_count < 2:
            log(
                "Error: classifier training needs at least 2 positive and 2 negative "
                "sorted PNGs in data/sorted/pos and data/sorted/neg."
            )
            return

        train_samples, val_samples = split_classifier_samples(samples, val_fraction=0.2, seed=42)
        log(
            f"Using sorted folders directly for classification: "
            f"{len(train_samples)} train / {len(val_samples)} val "
            f"({pos_count} pos, {neg_count} neg total)."
        )

        train_transform = build_classifier_transforms(self.imgsz, train=True)
        val_transform = build_classifier_transforms(self.imgsz, train=False)
        train_ds = ImagePathDataset(train_samples, transform=train_transform)
        val_ds = ImagePathDataset(val_samples, transform=val_transform)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=max(1, self.batch),
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, self.batch),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        model, pretrained_loaded, warning = create_classifier_model(self.size, pretrained=True)
        if warning:
            log(warning)
        log(
            f"Loading {format_backbone_label(self.size)} "
            f"({'pretrained' if pretrained_loaded else 'random init'})..."
        )
        model.to(device)

        train_pos = sum(1 for _, label in train_samples if label == 1)
        train_neg = sum(1 for _, label in train_samples if label == 0)
        total_train = max(1, train_pos + train_neg)
        class_weights = torch.tensor(
            [
                total_train / max(1, 2 * train_neg),
                total_train / max(1, 2 * train_pos),
            ],
            dtype=torch.float32,
            device=device,
        )

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        best_f1 = -1.0
        best_loss = math.inf
        best_state = None
        best_metrics = {}
        patience = 10
        stale_epochs = 0
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        log(
            f"Starting {self.arch} training with {format_backbone_label(self.size)} "
            f"(epochs={self.epochs}, imgsz={self.imgsz}, batch={self.batch})..."
        )

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for images, labels, _paths in train_loader:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                train_loss_sum += float(loss.item()) * batch_size
                train_correct += int((logits.argmax(dim=1) == labels).sum().item())
                train_total += batch_size

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            tp = fp = fn = 0

            with torch.no_grad():
                for images, labels, _paths in val_loader:
                    images = images.to(device, non_blocking=pin_memory)
                    labels = labels.to(device, non_blocking=pin_memory)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)

                    batch_size = labels.size(0)
                    val_loss_sum += float(loss.item()) * batch_size
                    val_correct += int((preds == labels).sum().item())
                    val_total += batch_size

                    tp += int(((preds == 1) & (labels == 1)).sum().item())
                    fp += int(((preds == 1) & (labels == 0)).sum().item())
                    fn += int(((preds == 0) & (labels == 1)).sum().item())

            train_loss = train_loss_sum / max(1, train_total)
            val_loss = val_loss_sum / max(1, val_total)
            train_acc = train_correct / max(1, train_total)
            val_acc = val_correct / max(1, val_total)
            val_precision = tp / max(1, tp + fp)
            val_recall = tp / max(1, tp + fn)
            val_f1 = 0.0
            if val_precision + val_recall > 0:
                val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)

            scheduler.step(val_f1)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)

            log(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}  precision={val_precision:.3f}  "
                f"recall={val_recall:.3f}  f1={val_f1:.3f}"
            )

            improved = (val_f1 > best_f1 + 1e-6) or (
                abs(val_f1 - best_f1) <= 1e-6 and val_loss < best_loss
            )
            if improved:
                best_f1 = val_f1
                best_loss = val_loss
                stale_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                    "pretrained_loaded": pretrained_loaded,
                }
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    log(f"Early stopping after {epoch} epochs with no F1 improvement.")
                    break

        if best_state is None:
            log("Training aborted before a usable checkpoint was created.")
            return

        model.load_state_dict(best_state)
        os.makedirs(self.run_dir, exist_ok=True)
        best_path = os.path.join(self.run_dir, "best_classifier.pth")
        save_classifier_checkpoint(
            best_path,
            model,
            backbone_key=self.size,
            imgsz=self.imgsz,
            extra={
                "metrics": best_metrics,
                "auto_positive_threshold": DEFAULT_AUTO_POSITIVE_THRESHOLD,
                "review_threshold": DEFAULT_REVIEW_THRESHOLD,
            },
        )
        save_classifier_checkpoint(
            os.path.join(self.run_dir, "last_classifier.pth"),
            model,
            backbone_key=self.size,
            imgsz=self.imgsz,
            extra={
                "metrics": best_metrics,
                "auto_positive_threshold": DEFAULT_AUTO_POSITIVE_THRESHOLD,
                "review_threshold": DEFAULT_REVIEW_THRESHOLD,
            },
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
        axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc")
        axes[1].plot(history["epoch"], history["val_recall"], label="Val Recall")
        axes[1].plot(history["epoch"], history["val_f1"], label="Val F1")
        axes[1].set_title("Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.suptitle(f"{format_backbone_label(self.size)} Classification")
        fig.tight_layout()
        plot_path = os.path.join(self.run_dir, "results.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        self.results_plot_path = plot_path

        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        ModelRegistry.register(self._make_registry_entry())
        log(f"\nTraining complete! Best model: {best_path}")

    def _make_registry_entry(self):
        run_dir = os.path.join(self.output_dir, self.run_name)
        # Ultralytics saves to weights/best.pt; RF-DETR saves checkpoint_best_ema.pth at run root
        if self.backend == "rfdetr":
            weights = self._find_rfdetr_best(run_dir)
        elif self.backend == "torchvision":
            weights = os.path.join(run_dir, "best_classifier.pth")
        elif self.backend == "nnunet":
            weights = os.path.join(
                "output", "nnunet_results", NNUNET_DATASET_NAME,
                f"nnUNetTrainer__nnUNetPlans__{self.size}",
                "fold_0", "checkpoint_best.pth",
            )
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

    # ── nnU-Net backend ───────────────────────────────────────────

    @staticmethod
    def _find_nnunet_cmd(name: str) -> str:
        """Locate an nnU-Net v2 CLI entry point, raising FileNotFoundError if absent."""
        import shutil
        found = shutil.which(name)
        if found:
            return found
        # Also search the same scripts directory as the active interpreter
        candidate = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"nnU-Net CLI '{name}' not found.\n"
            f"Install with:  pip install nnunetv2"
        )

    def _train_nnunet(self, log):
        """Train a nnU-Net v2 self-configuring U-Net on the fabella dataset."""
        import subprocess

        # Locate CLI tools
        try:
            plan_cmd  = self._find_nnunet_cmd("nnUNetv2_plan_and_preprocess")
            train_cmd = self._find_nnunet_cmd("nnUNetv2_train")
        except FileNotFoundError as err:
            log(str(err))
            return

        # Build directory paths and nnU-Net environment variables
        try:
            base_parent = os.path.dirname(os.path.dirname(os.path.normpath(self.pos_dir)))
        except Exception:
            base_parent = "data"
        nnunet_base      = os.path.abspath(os.path.join(base_parent, "nnunet"))
        raw_dir          = os.path.join(nnunet_base, "nnUNet_raw")
        preprocessed_dir = os.path.join(nnunet_base, "nnUNet_preprocessed")
        results_dir      = os.path.abspath(os.path.join("output", "nnunet_results"))
        env = {
            **os.environ,
            "nnUNet_raw":          raw_dir,
            "nnUNet_preprocessed": preprocessed_dir,
            "nnUNet_results":      results_dir,
            # Allow Python to find the custom trainer in the project root
            "PYTHONPATH": os.pathsep.join(
                filter(None, [os.path.abspath("."), os.environ.get("PYTHONPATH", "")])
            ),
        }

        dataset_dir = os.path.join(raw_dir, NNUNET_DATASET_NAME)

        # Step 1: Convert YOLO polygon labels → nnU-Net dataset format
        log("Preparing nnU-Net dataset from YOLO segmentation labels…")
        n_cases = self._prepare_nnunet_dataset(dataset_dir, base_parent, log)
        if n_cases == 0:
            log("Error: no labeled training images found. Label positive images first.")
            return

        # Step 2: Plan and preprocess using ResEncM planner (recommended over old default)
        # ResEncM is tailored for medium GPU budgets (RTX 4070 12 GB) and gives a larger
        # receptive field with a residual encoder — better for small objects.
        PLANS_ID = "nnUNetResEncUNetMPlans"
        log(f"\n[nnU-Net] Planning and preprocessing {n_cases} cases "
            f"(planner: nnUNetPlannerResEncM, configuration: {self.size})…")
        result = subprocess.run(
            [plan_cmd, "-d", str(NNUNET_DATASET_ID),
             "--verify_dataset_integrity",
             "-pl", "nnUNetPlannerResEncM",
             "-c", self.size],
            text=True, capture_output=True, env=env,
        )
        for line in (result.stdout or "").splitlines()[-80:]:
            log(line)
        if result.returncode != 0:
            log(f"[nnU-Net] Preprocessing failed:\n{result.stderr[-2000:]}")
            return

        # Step 3: Patch plans JSON for small-object detection
        patch_size = max(int(self.imgsz), 512)
        plans_path = os.path.join(
            preprocessed_dir, NNUNET_DATASET_NAME, f"{PLANS_ID}.json"
        )
        if os.path.exists(plans_path):
            try:
                import json as _json
                with open(plans_path, encoding="utf-8") as f:
                    plans = _json.load(f)
                config_key = self.size
                if config_key in plans.get("configurations", {}):
                    cfg = plans["configurations"][config_key]
                    original_patch = cfg.get("patch_size", [])
                    if len(original_patch) == 2:
                        cfg["patch_size"] = [patch_size, patch_size]
                    elif len(original_patch) == 3:
                        cfg["patch_size"] = [patch_size, patch_size, patch_size]
                    cfg["batch_size"] = max(cfg.get("batch_size", 2), 16)
                    plans["configurations"][config_key] = cfg
                    with open(plans_path, "w", encoding="utf-8") as f:
                        _json.dump(plans, f, indent=4)
                    log(f"[nnU-Net] Patched {PLANS_ID}.json: "
                        f"patch_size={patch_size}\u00d7{patch_size}, "
                        f"batch_size={cfg['batch_size']}")
                else:
                    log(f"[nnU-Net] Warning: config '{config_key}' not found in plans — using nnU-Net defaults.")
            except Exception as patch_err:
                log(f"[nnU-Net] Warning: could not patch plans file: {patch_err}")
        else:
            log(f"[nnU-Net] Warning: plans file not found at {plans_path} — using nnU-Net defaults.")

        # Step 4: Train fold 0 with custom Fabella trainer
        # -tr flag (not --trainer) is the correct nnUNetv2_train CLI syntax
        log(f"\n[nnU-Net] Training '{self.size}' — fold 0 with nnUNetTrainerFabella…")
        log(f"  CLI: nnUNetv2_train {NNUNET_DATASET_ID} {self.size} 0 "
            f"-tr nnUNetTrainerFabella -p {PLANS_ID} --npz")
        result = subprocess.run(
            [train_cmd, str(NNUNET_DATASET_ID), self.size, "0",
             "-tr", "nnUNetTrainerFabella",
             "-p", PLANS_ID,
             "--npz"],
            text=True, capture_output=True, env=env,
        )
        for line in (result.stdout or "").splitlines()[-120:]:
            log(line)
        if result.returncode != 0:
            log(f"[nnU-Net] Training failed:\n{result.stderr[-2000:]}")
            return

        log("\n[nnU-Net] Training complete!")
        ModelRegistry.register(self._make_registry_entry())

    def _prepare_nnunet_dataset(self, dataset_dir: str, base_parent: str, log) -> int:
        """Convert YOLO seg labels + sorted PNG images to nnU-Net v2 dataset layout.

        Directory layout produced::

            dataset_dir/
              dataset.json          ← channel/label metadata
              case_id_map.json      ← maps nnU-Net case IDs back to original filenames
              imagesTr/
                fabella_0000_0000.png   ← channel-0 grayscale X-ray
                ...
              labelsTr/
                fabella_0000.png        ← binary mask (0=background, 1=fabella)
                ...

        Returns the number of training cases written.
        """
        import cv2
        import json
        import numpy as np

        images_tr = os.path.join(dataset_dir, "imagesTr")
        labels_tr = os.path.join(dataset_dir, "labelsTr")
        os.makedirs(images_tr, exist_ok=True)
        os.makedirs(labels_tr, exist_ok=True)

        label_dir = os.path.join(base_parent, "labels", "seg")
        id_map: dict = {}
        count = 0

        for fname in sorted(os.listdir(self.pos_dir)):
            if not fname.lower().endswith(".png"):
                continue
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(label_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue

            img = cv2.imread(os.path.join(self.pos_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape

            # Convert YOLO normalised polygon to filled binary mask
            mask = np.zeros((h, w), dtype=np.uint8)
            with open(label_path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    coords = [float(v) for v in parts[1:]]
                    pts = np.array(
                        [(int(coords[i] * w), int(coords[i + 1] * h))
                         for i in range(0, len(coords) - 1, 2)],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(mask, [pts], 1)

            case_id = f"fabella_{count:04d}"
            cv2.imwrite(os.path.join(images_tr, f"{case_id}_0000.png"), img)
            cv2.imwrite(os.path.join(labels_tr, f"{case_id}.png"), mask)
            id_map[case_id] = fname
            count += 1

        if count == 0:
            return 0

        dataset_json = {
            "channel_names": {"0": "X-ray"},
            "labels": {"background": 0, "fabella": 1},
            "numTraining": count,
            "file_ending": ".png",
        }
        with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)
        # Store original filename mapping for traceability
        with open(os.path.join(dataset_dir, "case_id_map.json"), "w") as f:
            json.dump(id_map, f, indent=2)

        log(f"  Prepared {count} training cases → {images_tr}")
        return count
