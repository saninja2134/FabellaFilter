# SAM3-Assisted Auto-Labeler with Active Learning.
# Runs SAM3 in segment-everything mode, then scores/filters masks against
# features learned from existing labels.  No prompts guide SAM — references
# are used ONLY for ranking, so SAM discovers all structures before we pick.
# Roboflow-inspired workflow: predict → review → approve/reject → learn.
import cv2
import os
import math
import numpy as np


class SAM3AutoLabeler:
    """SAM3-based assisted labeling with active learning.

    Workflow
    --------
    1. Loads all existing seg labels and extracts a **feature fingerprint**
       for each one (area, aspect ratio, compactness, circularity, position).
    2. For each unlabeled image:
       a. Optionally crops to region-of-interest (from references) so tiny
          objects fill more of SAM's 1024 × 1024 internal resolution.
       b. Runs SAM in **segment-everything** mode — no prompts, no bias.
       c. Extracts features from every candidate mask and scores against
          the reference cluster using normalised feature distance.
       d. Shows the top candidates ranked by similarity.
       e. User **approves / rejects / manually edits**.
    3. Approved labels feed back into the reference pool — improving
       ranking for subsequent images.
    """

    # ── construction ──────────────────────────────────────────────

    def __init__(self, image_dir="data/sorted/pos", label_dir="data/labels/seg",
                 sam_model_path="sam3.pt"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sam_model_path = sam_model_path

        # Image list
        if os.path.exists(image_dir):
            self.images = sorted(
                f for f in os.listdir(image_dir) if f.lower().endswith('.png')
            )
        else:
            self.images = []

        os.makedirs(label_dir, exist_ok=True)

        # ── Reference feature pool ────────────────────────────────
        # Each entry: [area, aspect_ratio, compactness, cx, cy]
        self.ref_features: list[np.ndarray] = []

        # Current-image state
        self.index = 0
        self.current_image = None           # raw loaded image
        self.display_src = None             # 8-bit BGR for drawing
        self.proposed_polygons: list[dict] = []   # SAM candidates
        self.selected_idx = 0               # which candidate is highlighted
        self.approved_polygons: list[list[tuple[float, float]]] = []
        self.manual_polygon: list[tuple[float, float]] = []
        self.mode = "auto"                  # "auto" | "manual"

        # Zoom / pan
        self.zoom_level = 1.0
        self.offset = [50, 50]
        self.dragging = False
        self.last_mouse = [0, 0]

        # Session statistics
        self.stats = {"approved": 0, "rejected": 0, "edited": 0}

        self.window_name = (
            "SAM3 Auto-Labeler  |  Y/Space: Approve  |  N: Reject  |  "
            "E: Edit  |  Tab: Cycle  |  A/D: Nav  |  Q: Quit"
        )

        self.sam_model = None  # loaded lazily in run()

    # ── feature extraction ────────────────────────────────────────

    @staticmethod
    def _poly_features(pts_norm):
        """Compute a 5-D feature vector from normalised polygon points.

        Returns np.array([area, aspect_ratio, compactness, cx, cy])
        """
        n = len(pts_norm)
        if n < 3:
            return None

        xs = [p[0] for p in pts_norm]
        ys = [p[1] for p in pts_norm]
        cx = sum(xs) / n
        cy = sum(ys) / n

        # Shoelace area
        area = abs(sum(
            pts_norm[i][0] * pts_norm[(i + 1) % n][1]
            - pts_norm[(i + 1) % n][0] * pts_norm[i][1]
            for i in range(n)
        )) / 2.0

        # Bounding box aspect ratio
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        aspect = bw / bh if bh > 1e-8 else 1.0

        # Compactness = perimeter² / (4π · area) — 1.0 for a circle
        perimeter = sum(
            math.hypot(pts_norm[(i + 1) % n][0] - pts_norm[i][0],
                       pts_norm[(i + 1) % n][1] - pts_norm[i][1])
            for i in range(n)
        )
        compactness = (perimeter ** 2) / (4 * math.pi * area) if area > 1e-10 else 50.0

        return np.array([area, aspect, compactness, cx, cy], dtype=np.float64)

    # ── reference label loading ───────────────────────────────────

    def _load_references(self):
        """Scan all existing label files and extract feature vectors."""
        self.ref_features.clear()

        if not os.path.exists(self.label_dir):
            return

        for fname in os.listdir(self.label_dir):
            if not fname.endswith('.txt'):
                continue
            try:
                with open(os.path.join(self.label_dir, fname)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7:
                            continue
                        coords = [float(p) for p in parts[1:]]
                        pts = [(coords[i], coords[i + 1])
                               for i in range(0, len(coords), 2)]
                        feat = self._poly_features(pts)
                        if feat is not None:
                            self.ref_features.append(feat)
            except Exception:
                continue

        n = len(self.ref_features)
        if n:
            areas = [f[0] for f in self.ref_features]
            print(f"[SAM3] Loaded {n} reference(s).  "
                  f"Avg area: {np.mean(areas):.5f}  "
                  f"Range: [{min(areas):.5f}, {max(areas):.5f}]")
        else:
            print("[SAM3] No references found — first images will show all SAM masks unranked.")

    def _add_reference(self, pts_norm):
        """Register a single polygon (normalised) in the reference pool."""
        feat = self._poly_features(pts_norm)
        if feat is not None:
            self.ref_features.append(feat)

    # ── mask scoring ──────────────────────────────────────────────

    def _ref_mean_std(self):
        """Return (mean, std) of reference features, with safe minimums."""
        if not self.ref_features:
            return None, None
        arr = np.array(self.ref_features)
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0)
        sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
        return mu, sigma

    def _score_mask(self, pts_norm):
        """Score a candidate polygon (0 – 1) against reference feature cluster.

        Uses normalised Euclidean distance across all 5 features, mapped
        through exp(-d) so identical = 1.0, distant = 0.0.
        """
        if not self.ref_features:
            return 0.5  # no references — neutral score

        feat = self._poly_features(pts_norm)
        if feat is None:
            return 0.0

        mu, sigma = self._ref_mean_std()
        if mu is None:
            return 0.5

        # Weighted feature importance:
        #   area (0.35) + aspect (0.10) + compactness (0.15) + cx (0.20) + cy (0.20)
        weights = np.array([0.35, 0.10, 0.15, 0.20, 0.20])
        diff = np.abs(feat - mu) / sigma
        dist = float(np.sum(weights * diff))
        return float(np.exp(-0.5 * dist))

    # ── region-of-interest cropping ───────────────────────────────

    def _get_roi(self, img_h, img_w):
        """If references exist, return a pixel-space crop box that focuses
        on the likely region — gives SAM much higher effective resolution
        for small targets.  Returns (x1, y1, x2, y2) or None."""
        if len(self.ref_features) < 2:
            return None

        cxs = [f[3] for f in self.ref_features]
        cys = [f[4] for f in self.ref_features]
        areas = [f[0] for f in self.ref_features]

        avg_cx, avg_cy = float(np.mean(cxs)), float(np.mean(cys))
        # Estimate object radius from average area (treat as circle)
        avg_radius = float(np.sqrt(np.mean(areas) / math.pi))
        # Crop to 8× the object radius for generous context
        margin = max(avg_radius * 8, 0.15)

        x1 = int(max(0, (avg_cx - margin)) * img_w)
        y1 = int(max(0, (avg_cy - margin)) * img_h)
        x2 = int(min(1, (avg_cx + margin)) * img_w)
        y2 = int(min(1, (avg_cy + margin)) * img_h)

        # Only crop if the ROI is meaningfully smaller than the full image
        roi_area = (x2 - x1) * (y2 - y1)
        if roi_area < 0.7 * img_h * img_w and roi_area > 100:
            return (x1, y1, x2, y2)
        return None

    # ── mask utilities ────────────────────────────────────────────

    @staticmethod
    def _mask_to_polygon(binary_mask):
        """Convert binary mask → list of (x, y) pixel-coordinate tuples."""
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return []
        eps = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)
        return [(float(p[0][0]), float(p[0][1])) for p in approx]

    @staticmethod
    def _polygon_iou(poly1, poly2):
        """Approximate IoU between two normalised polygons (bbox-based)."""
        def _bb(pts):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return min(xs), min(ys), max(xs), max(ys)
        b1, b2 = _bb(poly1), _bb(poly2)
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0

    # ── SAM prediction ────────────────────────────────────────────

    def _prep_for_sam(self, img_path):
        """Load and normalise image to uint8 BGR for SAM."""
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        img8 = (raw / 256).astype(np.uint8) if raw.dtype == np.uint16 else raw.astype(np.uint8)
        img8 = cv2.normalize(img8, img8, 0, 255, cv2.NORM_MINMAX)
        if len(img8.shape) == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        elif img8.shape[2] == 1:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        return img8

    def _predict(self, img_path):
        """Run SAM3 segment-everything, then score & rank by reference similarity."""
        if self.current_image is None or self.sam_model is None:
            return []
        h, w = self.current_image.shape[:2]
        img8 = self._prep_for_sam(img_path)
        if img8 is None:
            return []

        # ── Optional ROI crop for higher effective resolution ─────
        roi = self._get_roi(h, w)
        if roi:
            rx1, ry1, rx2, ry2 = roi
            crop = img8[ry1:ry2, rx1:rx2]
            print(f"[SAM3]   Cropped to ROI {rx2-rx1}x{ry2-ry1}  "
                  f"(full image {w}x{h})")
        else:
            crop = img8
            rx1, ry1 = 0, 0

        candidates: list[dict] = []

        try:
            # Segment-everything: no prompts, SAM finds all masks
            results = self.sam_model(crop)
            for r in results:
                if r.masks is None:
                    continue
                for md in r.masks.data:
                    mask_np = md.cpu().numpy()
                    poly_crop = self._mask_to_polygon(mask_np)
                    if len(poly_crop) < 3:
                        continue

                    # Map crop coords back to full image then normalise
                    poly_full = [(px + rx1, py + ry1) for px, py in poly_crop]
                    poly_norm = [(px / w, py / h) for px, py in poly_full]

                    score = self._score_mask(poly_norm)
                    candidates.append({
                        "polygon": poly_full,
                        "polygon_norm": poly_norm,
                        "score": score,
                        "source": "auto",
                    })
        except Exception as e:
            print(f"[SAM3] prediction error: {e}")

        # Sort by score descending, deduplicate overlapping masks
        candidates.sort(key=lambda c: c["score"], reverse=True)
        filtered: list[dict] = []
        for c in candidates:
            if all(self._polygon_iou(c["polygon_norm"], f["polygon_norm"]) < 0.5
                   for f in filtered):
                filtered.append(c)
        return filtered[:8]

    # ── label I/O ─────────────────────────────────────────────────

    def _save_label(self, polygons_norm):
        """Save approved polygons in YOLO seg format and update reference pool."""
        if not polygons_norm or self.current_image is None:
            return
        txt = os.path.splitext(self.images[self.index])[0] + ".txt"
        with open(os.path.join(self.label_dir, txt), 'w') as f:
            for poly in polygons_norm:
                flat = []
                for x, y in poly:
                    flat.append(max(0.0, min(1.0, x)))
                    flat.append(max(0.0, min(1.0, y)))
                f.write(f"0 {' '.join(f'{v:.6f}' for v in flat)}\n")
        # Active learning: add to reference feature pool
        for poly in polygons_norm:
            self._add_reference(poly)
        print(f"[SAM3] Saved & learned: {txt}  (refs: {len(self.ref_features)})")

    def _has_label(self, img_name):
        txt = os.path.splitext(img_name)[0] + ".txt"
        return os.path.exists(os.path.join(self.label_dir, txt))

    def _get_unlabeled(self):
        return [f for f in self.images if not self._has_label(f)]

    # ── display / drawing ─────────────────────────────────────────

    def _to_screen(self, ix, iy):
        return (int(ix * self.zoom_level + self.offset[0]),
                int(iy * self.zoom_level + self.offset[1]))

    def _to_image(self, sx, sy):
        z = self.zoom_level if self.zoom_level > 0 else 0.001
        return ((sx - self.offset[0]) / z, (sy - self.offset[1]) / z)

    def mouse_callback(self, event, x, y, flags, param):
        img_x, img_y = self._to_image(x, y)

        if self.mode == "manual" and event == cv2.EVENT_LBUTTONDOWN:
            self.manual_polygon.append((img_x, img_y))
            self.redraw()
        elif self.mode == "manual" and event == cv2.EVENT_RBUTTONDOWN:
            if len(self.manual_polygon) >= 3:
                self.approved_polygons.append(self.manual_polygon.copy())
                self.manual_polygon = []
                self.redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.last_mouse = [x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.offset[0] += x - self.last_mouse[0]
            self.offset[1] += y - self.last_mouse[1]
            self.last_mouse = [x, y]
            self.redraw()
        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            sf = 1.15 if flags > 0 else 1.0 / 1.15
            self.offset[0] = x - (x - self.offset[0]) * sf
            self.offset[1] = y - (y - self.offset[1]) * sf
            self.zoom_level *= sf
            self.redraw()

    def redraw(self):
        if self.display_src is None:
            return
        canvas_h, canvas_w = 950, 1400
        display = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        h, w = self.display_src.shape[:2]
        sw = int(w * self.zoom_level)
        sh = int(h * self.zoom_level)

        if sw > 0 and sh > 0:
            resized = cv2.resize(self.display_src, (sw, sh))
            y1, y2 = max(0, int(self.offset[1])), min(canvas_h, int(self.offset[1]) + sh)
            x1, x2 = max(0, int(self.offset[0])), min(canvas_w, int(self.offset[0]) + sw)
            iy1 = max(0, -int(self.offset[1]))
            ix1 = max(0, -int(self.offset[0]))
            iy2 = iy1 + (y2 - y1)
            ix2 = ix1 + (x2 - x1)
            if y2 > y1 and x2 > x1 and iy2 > iy1 and ix2 > ix1:
                display[y1:y2, x1:x2] = resized[iy1:iy2, ix1:ix2]

        overlay = display.copy()

        # ── draw only the SELECTED SAM proposal ─────────────────
        if self.mode == "auto" and self.proposed_polygons:
            cand = self.proposed_polygons[self.selected_idx]
            score = cand["score"]

            # Colour: green (good) → yellow (maybe) → red (poor)
            if score >= 0.6:
                fill = (0, 200, 0)
                edge = (0, 255, 0)
            elif score >= 0.35:
                fill = (0, 200, 200)
                edge = (0, 255, 255)
            else:
                fill = (0, 0, 200)
                edge = (0, 0, 255)

            pts_scr = np.array(
                [self._to_screen(px, py) for px, py in cand["polygon"]],
                np.int32
            )
            cv2.fillPoly(overlay, [pts_scr], fill)
            cv2.polylines(display, [pts_scr], True, edge, 3)

            # Score label
            if len(pts_scr) > 0:
                tx, ty = pts_scr[0]
                tag = f"{score:.0%} ({cand['source']})"
                cv2.putText(display, tag, (tx, ty - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, edge, 2)

        # ── draw completed manual polygons (green, solid) ─────────
        for poly_px in self.approved_polygons:
            pts_scr = np.array(
                [self._to_screen(px, py) for px, py in poly_px], np.int32
            )
            cv2.fillPoly(overlay, [pts_scr], (0, 255, 0))
            cv2.polylines(display, [pts_scr], True, (0, 200, 0), 2)

        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

        # ── draw manual polygon in-progress ───────────────────────
        if self.mode == "manual" and self.manual_polygon:
            scr = []
            for px, py in self.manual_polygon:
                sx, sy = self._to_screen(px, py)
                scr.append([sx, sy])
                if 0 <= sx < canvas_w and 0 <= sy < canvas_h:
                    cv2.circle(display, (sx, sy), 4, (255, 0, 0), -1)
            if len(scr) > 1:
                cv2.polylines(display, [np.array(scr, np.int32)], False, (255, 0, 0), 2)

        # ── HUD with dark background bar ──────────────────────────
        unlabeled = self._get_unlabeled()
        remaining = len(unlabeled) - (self.queue_pos + 1) if hasattr(self, 'queue_pos') else "?"

        img_short = os.path.basename(self.images[self.index])[:40]
        hud_lines = [
            f"{img_short}  |  Left: {remaining}  |  Zoom: {self.zoom_level:.1f}x",
            f"Approved: {self.stats['approved']}  Rejected: {self.stats['rejected']}  Edited: {self.stats['edited']}  Refs: {len(self.ref_features)}",
        ]
        if self.mode == "auto":
            n_cand = len(self.proposed_polygons)
            sel = self.selected_idx + 1 if n_cand else 0
            hud_lines.append(
                f"AUTO  {sel}/{n_cand}  |  "
                "Y/Space=Approve  N=Reject  Tab=Next Mask  E=Manual"
            )
        else:
            n_poly = len(self.approved_polygons)
            hud_lines.append(
                f"MANUAL  ({n_poly} poly)  |  "
                "LClick=Point  RClick=Close  Space=Save All  Z=Undo  C=Clear  Esc=Back"
            )

        # Dark background bar
        bar_h = 12 + len(hud_lines) * 26
        display[0:bar_h, :] = (display[0:bar_h, :] * 0.25).astype(np.uint8)

        for i, line in enumerate(hud_lines):
            color = (0, 220, 255) if i == 0 else \
                    (180, 255, 180) if "AUTO" in line else \
                    (180, 180, 255) if "MANUAL" in line else (220, 220, 220)
            cv2.putText(display, line, (15, 24 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, display)

    def _draw_progress(self, current, total, img_name):
        """Draw a progress bar during the batch SAM prediction phase."""
        canvas_h, canvas_w = 950, 1400
        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Title
        cv2.putText(frame, "SAM3 — Processing All Images",
                    (canvas_w // 2 - 260, canvas_h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

        # Current file
        short = os.path.basename(img_name)[:60]
        cv2.putText(frame, short,
                    (canvas_w // 2 - 300, canvas_h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Progress bar
        bar_x, bar_y = 200, canvas_h // 2 + 10
        bar_w, bar_h = canvas_w - 400, 36
        pct = current / total if total > 0 else 0
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          (0, 200, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (120, 120, 120), 2)

        # Percentage + count
        pct_txt = f"{pct:.0%}  ({current}/{total})"
        cv2.putText(frame, pct_txt,
                    (canvas_w // 2 - 60, bar_y + bar_h + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)  # force repaint

    # ── main loop ─────────────────────────────────────────────────

    def run(self):
        """Entry point — call from main thread (uses OpenCV GUI)."""
        if not self.images:
            print(f"[SAM3] No images found in {self.image_dir}")
            return

        # Load SAM model
        print("[SAM3] Loading SAM model …")
        from ultralytics import SAM
        self.sam_model = SAM(self.sam_model_path)
        print("[SAM3] SAM model loaded.")

        # Build reference pool from existing labels
        self._load_references()

        # Work only on unlabeled images
        queue = self._get_unlabeled()
        if not queue:
            print("[SAM3] All images are already labeled!")
            return

        print(f"[SAM3] {len(queue)} unlabeled image(s) to process.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # ── PHASE 1: Batch predict ALL images with progress bar ───
        print("[SAM3] Starting batch prediction …")
        all_predictions: dict[str, list[dict]] = {}
        for i, img_name in enumerate(queue):
            self._draw_progress(i + 1, len(queue), img_name)

            img_path = os.path.join(self.image_dir, img_name)
            raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                all_predictions[img_name] = []
                continue

            # Store temporarily for _predict
            self.current_image = raw
            candidates = self._predict(img_path)
            all_predictions[img_name] = candidates

            n = len(candidates)
            top = f"{candidates[0]['score']:.0%}" if n else "—"
            print(f"[SAM3]  {i+1}/{len(queue)}  {img_name[:40]}  → {n} mask(s), top {top}")

        print(f"[SAM3] Batch prediction complete. Starting review …\n")

        # ── PHASE 2: Review loop (instant, no waiting) ────────────
        self.queue_pos = 0
        while self.queue_pos < len(queue):
            img_name = queue[self.queue_pos]
            self.index = self.images.index(img_name)
            img_path = os.path.join(self.image_dir, img_name)

            # Load image for display
            self.current_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if self.current_image is None:
                self.queue_pos += 1
                continue

            # Prepare 8-bit display
            disp = (self.current_image / 256).astype(np.uint8) \
                if self.current_image.dtype == np.uint16 else self.current_image
            dmin, dmax = disp.min(), disp.max()
            if dmax > dmin:
                disp = ((disp - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            if len(disp.shape) == 2:
                self.display_src = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            elif disp.shape[2] == 1:
                self.display_src = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            elif disp.shape[2] == 4:
                self.display_src = cv2.cvtColor(disp, cv2.COLOR_BGRA2BGR)
            else:
                self.display_src = disp

            # Reset per-image state
            self.mode = "auto"
            self.manual_polygon = []
            self.approved_polygons = []
            self.selected_idx = 0
            self.proposed_polygons = all_predictions.get(img_name, [])

            # Initial framing
            if self.zoom_level == 1.0:
                self.zoom_level, self.offset = 0.6, [200, 50]

            self.redraw()

            # Per-image key loop (instant — no SAM call here)
            action = self._handle_keys()
            if action == "quit":
                break

        cv2.destroyAllWindows()
        self._print_summary()

    def _handle_keys(self):
        """Block on key input and return 'next', 'quit', etc."""
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue  # no key pressed

            # ── MANUAL mode keys (checked first so Esc returns to auto) ──
            if self.mode == "manual":
                if key == 27:                            # Esc → back to auto
                    self.mode = "auto"
                    self.manual_polygon = []
                    self.approved_polygons = []
                    self.redraw()
                    continue

                if key == ord(' '):                      # Save manual labels
                    if self.approved_polygons and self.current_image is not None:
                        h, w = self.current_image.shape[:2]
                        norms = [
                            [(x / w, y / h) for x, y in poly]
                            for poly in self.approved_polygons
                        ]
                        self._save_label(norms)
                        self.stats["edited"] += 1
                    self.queue_pos += 1
                    return "next"

                if key == ord('c'):                      # Clear
                    self.manual_polygon = []
                    self.approved_polygons = []
                    self.redraw()

                if key == ord('z'):                      # Undo
                    if self.manual_polygon:
                        self.manual_polygon.pop()
                    elif self.approved_polygons:
                        self.approved_polygons.pop()
                    self.redraw()

                continue  # ignore other keys in manual mode

            # ── Quit (only in auto mode) ──────────────────────────
            if key == 27 or key == ord('q'):
                return "quit"

            # ── AUTO mode keys ────────────────────────────────────
            if self.mode == "auto":
                if key in (ord('y'), ord(' ')):         # Approve
                    if self.proposed_polygons:
                        cand = self.proposed_polygons[self.selected_idx]
                        self._save_label([cand["polygon_norm"]])
                        self.stats["approved"] += 1
                    self.queue_pos += 1
                    return "next"

                if key == ord('n'):                      # Reject
                    self.stats["rejected"] += 1
                    self.queue_pos += 1
                    return "next"

                if key == ord('\t') or key == 9:         # Tab — cycle candidates
                    if self.proposed_polygons:
                        self.selected_idx = (
                            (self.selected_idx + 1) % len(self.proposed_polygons)
                        )
                        self.redraw()

                if key == ord('e'):                       # Switch to manual
                    self.mode = "manual"
                    self.manual_polygon = []
                    self.approved_polygons = []
                    self.redraw()

                if key == ord('d'):                       # Skip forward
                    self.queue_pos += 1
                    return "next"
                if key == ord('a'):                       # Skip backward
                    self.queue_pos = max(0, self.queue_pos - 1)
                    return "next"

    # ── summary ───────────────────────────────────────────────────

    def _print_summary(self):
        total = self.stats["approved"] + self.stats["rejected"] + self.stats["edited"]
        print("\n" + "=" * 50)
        print("  SAM3 Auto-Labeler — Session Summary")
        print("=" * 50)
        print(f"  Approved (auto):  {self.stats['approved']}")
        print(f"  Rejected:         {self.stats['rejected']}")
        print(f"  Manually edited:  {self.stats['edited']}")
        print(f"  Total reviewed:   {total}")
        print(f"  Reference pool:   {len(self.ref_features)} labels")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    labeler = SAM3AutoLabeler()
    labeler.run()
