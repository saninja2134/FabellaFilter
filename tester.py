# Unified model tester supporting YOLO, RF-DETR, and classifier architectures.
import cv2
import numpy as np
import os
import shutil
from trainer import ARCHITECTURES, get_arch_info, get_rfdetr_class
from classifier_utils import (
    DEFAULT_AUTO_POSITIVE_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    load_classifier_checkpoint,
    predict_fabella_probability,
    load_png_bgr_for_overlay,
)


class ModelTester:
    # Architecture-aware inference runner.
    def __init__(self, arch="YOLO Seg", size="n",
                 src_dir="data/png/pos",
                 sorted_dir="data/sorted/pos"):
        info = get_arch_info(arch)
        self.arch       = arch
        self.size       = size
        self.backend    = info["backend"]
        self.task       = info["task"]
        self.src_dir    = src_dir
        self.sorted_dir = sorted_dir

        safe_arch = arch.lower().replace(" ", "_").replace("-", "_")
        self.run_name  = f"fabella_{safe_arch}_{size}_v1"
        if self.task == "classify":
            self.model_path = os.path.join("output", "runs", self.run_name, "best_classifier.pth")
        else:
            self.model_path = os.path.join("output/runs", self.run_name, "weights", "best.pt")
        self.output_dir = f"output/test_{safe_arch}_{size}"

    def run_test(self, progress_callback=None):
        def log(msg):
            print(msg)
            if progress_callback: progress_callback(msg)

        # Try .pt then .pth for RF-DETR checkpoints
        if not os.path.exists(self.model_path) and self.task != "classify":
            alt = self.model_path.replace(".pt", ".pth")
            if os.path.exists(alt):
                self.model_path = alt
        if not os.path.exists(self.model_path):
            log(f"Error: model not found at {self.model_path}")
            return

        # Gather unsorted test images
        if not os.path.exists(self.src_dir):
            log(f"Error: {self.src_dir} not found.")
            return

        all_images    = sorted([f for f in os.listdir(self.src_dir) if f.lower().endswith('.png')])
        sorted_images = set(os.listdir(self.sorted_dir)) if os.path.exists(self.sorted_dir) else set()
        unsorted      = [f for f in all_images if f not in sorted_images]

        if not unsorted:
            log("No unsorted images found for testing.")
            return

        log(f"Running {self.arch} inference on {len(unsorted)} unsorted images...")

        if self.backend == "torchvision":
            auto_dir = os.path.join(self.output_dir, "auto_positive")
            review_dir = os.path.join(self.output_dir, "review_band")
            remain_dir = os.path.join(self.output_dir, "remaining_manual")
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(auto_dir)
            os.makedirs(review_dir)
            os.makedirs(remain_dir)
            self._test_torchvision_classifier(
                unsorted,
                auto_dir,
                review_dir,
                remain_dir,
                log,
            )
            log(f"\nTest complete!")
            log(f"Auto-positive:    {len(os.listdir(auto_dir))}")
            log(f"Review band:      {len(os.listdir(review_dir))}")
            log(f"Remaining manual: {len(os.listdir(remain_dir))}")
            log(f"Results in: {self.output_dir}/")
            return

        det_dir   = os.path.join(self.output_dir, "detected")
        undet_dir = os.path.join(self.output_dir, "undetected")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(det_dir)
        os.makedirs(undet_dir)

        if self.backend == "ultralytics":
            self._test_ultralytics(unsorted, det_dir, undet_dir, log)
        elif self.backend == "rfdetr":
            self._test_rfdetr(unsorted, det_dir, undet_dir, log)

        log(f"\nTest complete!")
        log(f"Detected:   {len(os.listdir(det_dir))}")
        log(f"Undetected: {len(os.listdir(undet_dir))}")
        log(f"Results in: {self.output_dir}/")

    def _test_torchvision_classifier(self, batch, auto_dir, review_dir, remain_dir, log):
        model, checkpoint, device, transform = load_classifier_checkpoint(self.model_path)
        auto_threshold = float(
            checkpoint.get("auto_positive_threshold", DEFAULT_AUTO_POSITIVE_THRESHOLD)
        )
        review_threshold = float(
            checkpoint.get("review_threshold", DEFAULT_REVIEW_THRESHOLD)
        )
        review_threshold = min(review_threshold, auto_threshold)
        log(
            f"Classifier triage thresholds: auto-positive >= {auto_threshold:.2f}, "
            f"review >= {review_threshold:.2f}"
        )

        for i, img_name in enumerate(batch):
            img_path = os.path.join(self.src_dir, img_name)
            img_visual = load_png_bgr_for_overlay(img_path)
            if img_visual is None:
                continue

            try:
                prob = predict_fabella_probability(model, transform, img_path, device)
            except Exception as exc:
                log(f"  Inference error on {img_name}: {exc}")
                prob = 0.0

            if prob >= auto_threshold:
                dst_dir = auto_dir
                label = "AUTO+"
                color = (0, 200, 0)
            elif prob >= review_threshold:
                dst_dir = review_dir
                label = "REVIEW"
                color = (0, 180, 255)
            else:
                dst_dir = remain_dir
                label = "MANUAL"
                color = (180, 180, 180)

            cv2.putText(
                img_visual,
                f"{label} fabella={prob * 100:.1f}%",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.imwrite(os.path.join(dst_dir, img_name), img_visual)

            if (i + 1) % 50 == 0:
                log(f"  Processed {i + 1}/{len(batch)}...")

    # ── BACKENDS ──────────────────────────────────────────────────

    def _prep_visual(self, img_path):
        # Load image and normalise to uint8 BGR for visualisation.
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.dtype == np.uint16:
            img8 = (raw / 256).astype(np.uint8)
        else:
            img8 = raw.astype(np.uint8)
        img8 = cv2.normalize(img8, None, 0, 255, cv2.NORM_MINMAX)
        if len(img8.shape) == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        return img8

    def _save(self, img, has_det, fname, det_dir, undet_dir):
        dst = os.path.join(det_dir if has_det else undet_dir, fname)
        cv2.imwrite(dst, img)

    def _test_ultralytics(self, batch, det_dir, undet_dir, log):
        from ultralytics import YOLO
        model = YOLO(self.model_path)

        for i, img_name in enumerate(batch):
            img_path   = os.path.join(self.src_dir, img_name)
            img_visual = self._prep_visual(img_path)
            if img_visual is None:
                continue

            results = model.predict(source=img_path, conf=0.25, imgsz=1024, verbose=False)
            result  = results[0]
            has_det = False

            if self.task == "obb":
                if hasattr(result, "obb") and result.obb is not None and len(result.obb.xyxyxyxy) > 0:
                    has_det = True
                    for box, conf in zip(result.obb.xyxyxyxy, result.obb.conf):
                        pts = box.cpu().numpy().astype(np.int32)
                        cv2.polylines(img_visual, [pts], True, (0, 255, 0), 2)
                        self._draw_label(img_visual, pts, f"{conf:.2f}")

            elif self.task == "segment":
                if hasattr(result, "masks") and result.masks is not None and len(result.masks.xy) > 0:
                    has_det = True
                    overlay = img_visual.copy()
                    for mask, conf in zip(result.masks.xy, result.boxes.conf):
                        pts = mask.astype(np.int32)
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.polylines(img_visual, [pts], True, (0, 200, 0), 2)
                        self._draw_label(img_visual, pts, f"{conf:.2f}")
                    cv2.addWeighted(overlay, 0.3, img_visual, 0.7, 0, img_visual)

            else:  # detect (RT-DETR)
                if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                    has_det = True
                    for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_visual, f"{conf:.2f}", (x2 + 5, y1 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self._save(img_visual, has_det, img_name, det_dir, undet_dir)
            if (i + 1) % 50 == 0:
                log(f"  Processed {i + 1}/{len(batch)}...")

    def _test_rfdetr(self, batch, det_dir, undet_dir, log):
        try:
            cls_name, model_cls = get_rfdetr_class(self.task, self.size)
        except ValueError as e:
            log(f"Error: {e}")
            return
        except ImportError:
            log("Error: rfdetr package not installed. Run: pip install rfdetr")
            return
        except AttributeError:
            log(f"Error: RF-DETR class not found. For XL/2XL: pip install rfdetr[plus]")
            return

        log(f"Loading {cls_name} from {self.model_path}...")
        try:
            model = model_cls(pretrain_weights=self.model_path)
        except Exception as e:
            log(f"Error loading finetuned weights: {e}")
            log("Falling back to default pretrained weights...")
            model = model_cls()

        from PIL import Image as PILImage
        for i, img_name in enumerate(batch):
            img_path   = os.path.join(self.src_dir, img_name)
            img_visual = self._prep_visual(img_path)
            if img_visual is None:
                continue

            try:
                pil_img  = PILImage.open(img_path).convert("RGB")
                detects  = model.predict(pil_img, threshold=0.5)
                # rfdetr returns supervision.Detections with .xyxy, .confidence, .class_id
                # Seg variants also include .mask (N, H, W boolean arrays)
                has_det  = len(detects) > 0

                if has_det:
                    boxes = detects.xyxy                   # (N, 4) ndarray
                    confs = detects.confidence              # (N,) ndarray
                    masks = getattr(detects, 'mask', None)  # (N, H, W) or None

                    if masks is not None and len(masks) > 0:
                        # Draw segmentation masks + bounding boxes
                        overlay = img_visual.copy()
                        for mask, box, conf in zip(masks, boxes, confs):
                            # mask is a boolean (H, W) array
                            color = (0, 255, 0)
                            overlay[mask] = color
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(img_visual, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_visual, f"{conf:.2f}", (x2 + 5, y1 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.addWeighted(overlay, 0.3, img_visual, 0.7, 0, img_visual)
                    else:
                        # Detection-only: draw bounding boxes
                        for box, conf in zip(boxes, confs):
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_visual, f"{conf:.2f}", (x2 + 5, y1 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except Exception as e:
                log(f"  Inference error on {img_name}: {e}")
                has_det = False

            self._save(img_visual, has_det, img_name, det_dir, undet_dir)
            if (i + 1) % 50 == 0:
                log(f"  Processed {i + 1}/{len(batch)}...")

    def _draw_label(self, img, pts, text):
        if len(pts) == 0:
            return
        rx = int(np.max(pts[:, 0]))
        ry = int(np.mean(pts[:, 1]))
        cv2.putText(img, text, (rx + 15, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
