"""Tkinter shape-analysis workspace for fabella segmentation contours.

This module adds a contour-based statistical shape analysis workflow on top of
the 2D polygon labels stored in ``data/labels/seg``. The current repository
does not contain a true 3D mesh cohort, so the implementation explicitly works
with 2D outlines and reports 2D shape metrics.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import threading
import tkinter as tk
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import date
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.stats import f, gaussian_kde, linregress


STANDARD_LATERALITY = "Right"
RESAMPLE_POINTS = 64
PERMANOVA_PERMUTATIONS = 499
MISSING_VALUE_LABEL = "(Missing)"
NO_FILTER_LABEL = "No Filter"

METRIC_SPECS = OrderedDict(
    [
        ("area_px2", "Area (px^2)"),
        ("perimeter_px", "Perimeter (px)"),
        ("max_diameter_px", "Max Diameter (px)"),
        ("width_px", "Width (px)"),
        ("height_px", "Height (px)"),
        ("aspect_ratio", "Aspect Ratio"),
        ("compactness", "Compactness"),
        ("normalized_area", "Normalized Area"),
        ("normalized_perimeter", "Normalized Perimeter"),
        ("centroid_size_px", "Centroid Size (px)"),
    ]
)
DEFAULT_DIMENSIONS = ["area_px2", "perimeter_px", "max_diameter_px", "aspect_ratio"]

ICD_CONDITION_COLUMNS: List[str] = [
    "autoimmune",
    "diabetes",
    "hypertension",
    "joint_infection",
    "knee_osteoarthritis",
    "knee_osteomyelitis",
    "obesity",
    "nicotine_use",
    "trauma_lower_extremity",
]

ICD_DISPLAY_NAMES: Dict[str, str] = {
    "autoimmune": "Autoimmune",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "joint_infection": "Joint Infection",
    "knee_osteoarthritis": "Knee Osteoarthritis",
    "knee_osteomyelitis": "Knee Osteomyelitis",
    "obesity": "Obesity",
    "nicotine_use": "Nicotine Use",
    "trauma_lower_extremity": "Trauma Lower Extremity",
}

KNOWN_FACTOR_TYPES = {
    "Sex": "categorical",
    "Race": "categorical",
    "Ethnicity": "categorical",
    "Original Laterality": "categorical",
    "Weight Bearing": "categorical",
    "Arthroplasty": "categorical",
    "Mirror Applied": "categorical",
    "Age At Exam": "continuous",
    "Pain Score": "continuous",
    # ICD-derived binary conditions
    "Autoimmune": "categorical",
    "Diabetes": "categorical",
    "Hypertension": "categorical",
    "Joint Infection": "categorical",
    "Knee Osteoarthritis": "categorical",
    "Knee Osteomyelitis": "categorical",
    "Obesity": "categorical",
    "Nicotine Use": "categorical",
    "Trauma Lower Extremity": "categorical",
}


@dataclass
class ManualOverride:
    extra_rotation_deg: int = 0
    force_mirror: Optional[bool] = None


@dataclass
class RawShapeRecord:
    uid: str
    empi: Optional[str]
    study_date: Optional[str]
    study_date_obj: Optional[date]
    label_path: str
    image_path: Optional[str]
    image_size: Tuple[int, int]
    raw_points_px: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    factors: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RawDataset:
    records: List[RawShapeRecord]
    summary: Dict[str, Any]
    project_root: str
    emory_root: str


@dataclass
class ProcessedShapeRecord:
    raw: RawShapeRecord
    corrected_points_px: np.ndarray
    pre_procrustes_points: np.ndarray
    aligned_points: np.ndarray
    metrics: Dict[str, float]
    factor_values: Dict[str, Any]
    mirror_applied: bool
    auto_rotation_deg: float
    extra_rotation_deg: int
    final_rotation_deg: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class ShapeAnalysisResults:
    raw_dataset: RawDataset
    records: List[ProcessedShapeRecord]
    mean_shape: np.ndarray
    pca_scores: np.ndarray
    explained_variance_ratio: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    factor_types: Dict[str, str]
    categorical_factors: List[str]
    continuous_factors: List[str]
    default_pc_count: int


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def normalize_side(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower()
    if text in {"l", "left"}:
        return "Left"
    if text in {"r", "right"}:
        return "Right"
    if text in {"bilateral", "both", "knees"}:
        return "Bilateral"
    if "left" in text or text == "l knee":
        return "Left"
    if "right" in text or text == "r knee":
        return "Right"
    return None


def yes_no_label(value: Any) -> Optional[str]:
    numeric = safe_float(value)
    if numeric is None:
        return None
    return "Yes" if numeric > 0 else "No"


def arthroplasty_label(value: Any) -> Optional[str]:
    text = str(value).strip().upper()
    if not text or text == "0":
        return "None"
    if text == "L":
        return "Left"
    if text == "R":
        return "Right"
    return text


def format_value(value: Any) -> str:
    if value is None:
        return MISSING_VALUE_LABEL
    if isinstance(value, float):
        if math.isnan(value):
            return MISSING_VALUE_LABEL
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    return text if text else MISSING_VALUE_LABEL


def close_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return np.vstack([points, points[0]])


def polygon_area(points: np.ndarray) -> float:
    closed = close_points(points)
    return 0.5 * float(
        np.sum(closed[:-1, 0] * closed[1:, 1] - closed[:-1, 1] * closed[1:, 0])
    )


def polygon_perimeter(points: np.ndarray) -> float:
    closed = close_points(points)
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def pairwise_max_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(pdist(points).max())


def rotate_points(points: np.ndarray, angle_deg: float, center: Optional[np.ndarray] = None) -> np.ndarray:
    if center is None:
        center = points.mean(axis=0)
    theta = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )
    return (points - center) @ rotation.T + center


def mirror_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    mirrored = points.copy()
    mirrored[:, 0] = 2.0 * center[0] - mirrored[:, 0]
    return mirrored


def normalize_angle(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def auto_orient_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle_deg = math.degrees(math.atan2(major_axis[1], major_axis[0]))
    rotation_deg = -angle_deg
    rotated = rotate_points(points, rotation_deg)
    centered_rotated = rotated - rotated.mean(axis=0)
    if abs(centered_rotated[:, 0].max()) < abs(centered_rotated[:, 0].min()):
        rotated = rotate_points(rotated, 180.0)
        rotation_deg += 180.0
    return rotated, normalize_angle(rotation_deg)


def resample_closed_contour(points: np.ndarray, num_points: int = RESAMPLE_POINTS) -> np.ndarray:
    if len(points) < 3:
        return points.copy()
    closed = close_points(points)
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segment_lengths.sum())
    if perimeter <= 0:
        return points.copy()
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, cumulative[-1], num_points + 1)[:-1]

    # Vectorized 1D linear interpolation
    resampled_x = np.interp(targets, cumulative, closed[:, 0])
    resampled_y = np.interp(targets, cumulative, closed[:, 1])
    return np.column_stack([resampled_x, resampled_y])


def reanchor_contour(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    index = int(np.argmax(points[:, 0]))
    return np.roll(points, -index, axis=0)


def standardize_contour_winding(points: np.ndarray, target_sign: float = -1.0) -> np.ndarray:
    if len(points) < 3:
        return points
    area = polygon_area(points)
    if area == 0:
        return points
    if (area > 0 and target_sign < 0) or (area < 0 and target_sign > 0):
        reversed_points = points[::-1].copy()
        return reanchor_contour(reversed_points)
    return points


def centroid_size(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.sum(centered * centered)))


def orthogonal_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return source @ rotation


def generalized_procrustes(shapes: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    normalized_shapes = []
    for shape in shapes:
        centered = shape - shape.mean(axis=0)
        size = centroid_size(centered)
        normalized_shapes.append(centered / size if size > 0 else centered)
    aligned = np.asarray(normalized_shapes, dtype=float)
    mean_shape = aligned[0].copy()
    for _ in range(32):
        updated = []
        for shape in aligned:
            updated.append(orthogonal_align(shape, mean_shape))
        aligned = np.asarray(updated, dtype=float)
        new_mean = aligned.mean(axis=0)
        new_mean -= new_mean.mean(axis=0)
        size = centroid_size(new_mean)
        if size > 0:
            new_mean /= size
        if np.linalg.norm(new_mean - mean_shape) < 1e-7:
            mean_shape = new_mean
            break
        mean_shape = new_mean
    return aligned, mean_shape


def compute_shape_metrics(points: np.ndarray) -> Dict[str, float]:
    area = abs(polygon_area(points))
    perimeter = polygon_perimeter(points)
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    max_diameter = pairwise_max_distance(points)
    compactness = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else float("nan")
    normalized_area = (area / (max_diameter * max_diameter)) if max_diameter > 0 else float("nan")
    normalized_perimeter = (perimeter / max_diameter) if max_diameter > 0 else float("nan")
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "max_diameter_px": max_diameter,
        "width_px": width,
        "height_px": height,
        "aspect_ratio": (width / height) if height > 0 else float("nan"),
        "compactness": compactness,
        "normalized_area": normalized_area,
        "normalized_perimeter": normalized_perimeter,
        "centroid_size_px": centroid_size(points),
    }


def parse_segmentation_label(path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    warnings: List[str] = []
    polygons: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                values = [float(value) for value in stripped.split()]
            except ValueError:
                warnings.append(f"Non-numeric contour data on line {line_number}")
                continue
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                warnings.append(f"Malformed contour on line {line_number}")
                continue
            polygon = np.asarray(values[1:], dtype=float).reshape(-1, 2)
            polygons.append(polygon)
    if not polygons:
        return None, warnings
    if len(polygons) > 1:
        warnings.append(f"{len(polygons)} contours found; using the largest outline")
    largest = max(polygons, key=lambda polygon: abs(polygon_area(polygon)))
    return largest, warnings


def find_image_path(project_root: str, uid: str) -> Optional[str]:
    base_dirs = [
        os.path.join(project_root, "data", "sorted", "pos"),
        os.path.join(project_root, "data", "png", "pos"),
        os.path.join(project_root, "data", "coco", "train"),
        os.path.join(project_root, "data", "coco", "val"),
        os.path.join(project_root, "data", "coco", "test"),
    ]
    extensions = [".png", ".jpg", ".jpeg"]
    for directory in base_dirs:
        for extension in extensions:
            candidate = os.path.join(directory, uid + extension)
            if os.path.exists(candidate):
                return candidate
    return None


def read_image_size(image_path: Optional[str], metadata: Dict[str, Any]) -> Tuple[int, int]:
    if image_path:
        try:
            with Image.open(image_path) as image:
                return int(image.size[0]), int(image.size[1])
        except Exception:
            pass
    width = int(safe_float(metadata.get("img_width")) or 1)
    height = int(safe_float(metadata.get("img_height")) or 1)
    return max(width, 1), max(height, 1)


def load_image_metadata(image_metadata_path: str, segmentation_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, str]], int]:
    matched: Dict[str, Dict[str, str]] = {}
    total_rows = 0
    segmentation_id_set = set(segmentation_ids)
    if not os.path.exists(image_metadata_path):
        return matched, total_rows
    with open(image_metadata_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            uid = row.get("SOPInstanceUID_anon", "")
            if uid in segmentation_id_set:
                matched[uid] = row
    return matched, total_rows


def load_demographics(demographics_path: str, empis: Sequence[str]) -> Dict[str, Dict[str, str]]:
    matches: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(demographics_path):
        return matches
    wanted = set(empi for empi in empis if empi)
    with open(demographics_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            empi = row.get("empi_anon")
            if empi in wanted:
                matches[empi] = row
    return matches


def load_icd_conditions(icd_path: str, empis: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Return a dict mapping empi -> {display_name: 'Yes'/'No'} aggregated across all ICD rows.

    A condition is 'Yes' for a patient if any row for that patient has a non-zero value.
    """
    results: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(icd_path):
        return results
    wanted = set(empi for empi in empis if empi)
    # Pre-initialise all wanted patients to 'No' for every condition.
    for empi in wanted:
        results[empi] = {ICD_DISPLAY_NAMES[col]: "No" for col in ICD_CONDITION_COLUMNS}
    with open(icd_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            empi = row.get("empi_anon")
            if empi not in wanted:
                continue
            patient = results[empi]
            for col in ICD_CONDITION_COLUMNS:
                display = ICD_DISPLAY_NAMES[col]
                if patient[display] == "Yes":
                    continue  # already flagged
                raw = row.get(col, "0").strip()
                if raw and raw != "0":
                    patient[display] = "Yes"
    return results


def load_pain_scores(pain_path: str, empis: Sequence[str]) -> Dict[Tuple[str, str, str], float]:
    if not os.path.exists(pain_path):
        return {}
    wanted = set(empi for empi in empis if empi)
    accumulator: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    with open(pain_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            empi = row.get("empi_anon")
            if empi not in wanted:
                continue
            score = safe_float(row.get("pain_score"))
            if score is None:
                continue
            study_date = row.get("date_anon", "")
            laterality = normalize_side(row.get("laterality")) or normalize_side(row.get("pain_location"))
            if laterality == "Bilateral":
                accumulator[(empi, study_date, "Left")].append(score)
                accumulator[(empi, study_date, "Right")].append(score)
            elif laterality in {"Left", "Right"}:
                accumulator[(empi, study_date, laterality)].append(score)
    return {key: float(np.mean(values)) for key, values in accumulator.items()}


def build_factor_map(
    metadata: Dict[str, str],
    demographics: Dict[str, str],
    pain_lookup: Dict[Tuple[str, str, str], float],
    icd_lookup: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    empi = metadata.get("empi_anon")
    study_date = metadata.get("StudyDate_anon", "")
    laterality = normalize_side(metadata.get("laterality"))
    factors: Dict[str, Any] = {
        "Sex": demographics.get("sex"),
        "Race": demographics.get("race"),
        "Ethnicity": demographics.get("ethnicity"),
        "Original Laterality": laterality,
        "Weight Bearing": yes_no_label(metadata.get("weight_bearing")),
        "Arthroplasty": arthroplasty_label(metadata.get("arthroplasty")),
        "Age At Exam": safe_float(metadata.get("age_at_exam")),
        "Pain Score": pain_lookup.get((empi or "", study_date, laterality or "")),
    }
    icd_conditions = icd_lookup.get(empi or "", {})
    factors.update(icd_conditions)
    return factors


def load_raw_dataset(project_root: str, emory_root: str, log_callback: Optional[Any] = None) -> RawDataset:
    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    seg_dir = os.path.join(project_root, "data", "labels", "seg")
    image_metadata_path = os.path.join(emory_root, "MRKR_image_metadata.csv")
    demographics_path = os.path.join(emory_root, "MRKR_demographics.csv")
    pain_path = os.path.join(emory_root, "MRKR_pain.csv")
    icd_path = os.path.join(emory_root, "MRKR_ICD.csv")

    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    label_files = sorted(
        [
            os.path.join(seg_dir, filename)
            for filename in os.listdir(seg_dir)
            if filename.lower().endswith(".txt")
        ]
    )
    segmentation_ids = [os.path.splitext(os.path.basename(path))[0] for path in label_files]
    log(f"Shape analysis: scanning {len(label_files)} segmentation contours from data/labels/seg")

    image_metadata, total_image_rows = load_image_metadata(image_metadata_path, segmentation_ids)
    matched_empis = [row.get("empi_anon") for row in image_metadata.values() if row.get("empi_anon")]
    demographics_lookup = load_demographics(demographics_path, matched_empis)
    pain_lookup = load_pain_scores(pain_path, matched_empis)
    log("Shape analysis: loading ICD conditions (this may take a moment for large files)")
    icd_lookup = load_icd_conditions(icd_path, matched_empis)

    records: List[RawShapeRecord] = []
    summary = {
        "segmentation_count": len(label_files),
        "matched_image_metadata": 0,
        "matched_demographics": 0,
        "matched_pain": 0,
        "matched_icd": 0,
        "missing_image_files": 0,
        "unmatched_image_metadata_rows": max(total_image_rows - len(image_metadata), 0),
        "critical_warnings": [],
    }

    for label_path in label_files:
        uid = os.path.splitext(os.path.basename(label_path))[0]
        polygon_norm, label_warnings = parse_segmentation_label(label_path)
        if polygon_norm is None or len(polygon_norm) < 3:
            summary["critical_warnings"].append(f"{uid}: invalid contour label")
            continue

        metadata = image_metadata.get(uid, {})
        demographics = demographics_lookup.get(metadata.get("empi_anon", ""), {})
        image_path = find_image_path(project_root, uid)
        image_width, image_height = read_image_size(image_path, metadata)
        points_px = np.column_stack([polygon_norm[:, 0] * image_width, polygon_norm[:, 1] * image_height])

        warnings = list(label_warnings)
        if not metadata:
            warnings.append("No matching image metadata")
        if not demographics and metadata:
            warnings.append("No matching demographics record")
        if not image_path:
            warnings.append("Image file not found; using metadata dimensions")
            summary["missing_image_files"] += 1

        factors = build_factor_map(metadata, demographics, pain_lookup, icd_lookup) if metadata else {}
        if metadata:
            summary["matched_image_metadata"] += 1
        if demographics:
            summary["matched_demographics"] += 1
        if factors.get("Pain Score") is not None:
            summary["matched_pain"] += 1
        empi_key = metadata.get("empi_anon", "")
        if empi_key and empi_key in icd_lookup:
            summary["matched_icd"] += 1

        records.append(
            RawShapeRecord(
                uid=uid,
                empi=metadata.get("empi_anon") or None,
                study_date=metadata.get("StudyDate_anon") or None,
                study_date_obj=parse_iso_date(metadata.get("StudyDate_anon")),
                label_path=label_path,
                image_path=image_path,
                image_size=(image_width, image_height),
                raw_points_px=points_px,
                metadata=metadata,
                factors=factors,
                warnings=warnings,
            )
        )

    return RawDataset(records=records, summary=summary, project_root=project_root, emory_root=emory_root)


def collect_available_factors(records: Sequence[ProcessedShapeRecord]) -> Tuple[Dict[str, str], List[str], List[str]]:
    factor_types: Dict[str, str] = {}
    categorical: List[str] = []
    continuous: List[str] = []
    if not records:
        return factor_types, categorical, continuous

    keys = sorted(records[0].factor_values.keys())
    for key in keys:
        values = [record.factor_values.get(key) for record in records]
        cleaned = [value for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
        if not cleaned:
            continue
        inferred_type = KNOWN_FACTOR_TYPES.get(key)
        if inferred_type is None:
            if all(isinstance(value, (int, float, np.floating)) for value in cleaned):
                inferred_type = "continuous"
            else:
                inferred_type = "categorical"
        if inferred_type == "continuous":
            unique_numeric = {round(float(value), 6) for value in cleaned if safe_float(value) is not None}
            if len(unique_numeric) >= 3:
                factor_types[key] = "continuous"
                continuous.append(key)
        else:
            unique_values = sorted({format_value(value) for value in cleaned})
            if 2 <= len(unique_values) <= 12:
                factor_types[key] = "categorical"
                categorical.append(key)
    return factor_types, categorical, continuous


def process_dataset(dataset: RawDataset, manual_overrides: Dict[str, ManualOverride]) -> ShapeAnalysisResults:
    processed_records: List[ProcessedShapeRecord] = []
    pre_shapes: List[np.ndarray] = []

    for raw_record in dataset.records:
        override = manual_overrides.get(raw_record.uid, ManualOverride())
        points = raw_record.raw_points_px.copy()
        warnings = list(raw_record.warnings)

        original_laterality = raw_record.factors.get("Original Laterality")
        if override.force_mirror is None:
            mirror_applied = original_laterality == "Left"
        else:
            mirror_applied = bool(override.force_mirror)
            warnings.append("Manual mirror override applied")
        if mirror_applied:
            points = mirror_points(points)

        auto_corrected, auto_rotation_deg = auto_orient_points(points)
        corrected_points = auto_corrected.copy()
        if override.extra_rotation_deg:
            corrected_points = rotate_points(corrected_points, override.extra_rotation_deg)

        resampled = resample_closed_contour(corrected_points, RESAMPLE_POINTS)
        resampled -= resampled.mean(axis=0)
        resampled = reanchor_contour(resampled)
        # PDM averaging requires a consistent walk direction around the contour.
        resampled = standardize_contour_winding(resampled, target_sign=-1.0)

        factor_values = dict(raw_record.factors)
        factor_values["Mirror Applied"] = "Yes" if mirror_applied else "No"
        metrics = compute_shape_metrics(corrected_points)

        processed_records.append(
            ProcessedShapeRecord(
                raw=raw_record,
                corrected_points_px=corrected_points,
                pre_procrustes_points=resampled,
                aligned_points=np.zeros_like(resampled),
                metrics=metrics,
                factor_values=factor_values,
                mirror_applied=mirror_applied,
                auto_rotation_deg=auto_rotation_deg,
                extra_rotation_deg=int(override.extra_rotation_deg),
                final_rotation_deg=normalize_angle(auto_rotation_deg + float(override.extra_rotation_deg)),
                warnings=warnings,
            )
        )
        pre_shapes.append(resampled)

    if not processed_records:
        empty = np.zeros((0, 2))
        return ShapeAnalysisResults(
            raw_dataset=dataset,
            records=[],
            mean_shape=empty,
            pca_scores=np.zeros((0, 0)),
            explained_variance_ratio=np.zeros(0),
            eigenvalues=np.zeros(0),
            eigenvectors=np.zeros((0, 0)),
            factor_types={},
            categorical_factors=[],
            continuous_factors=[],
            default_pc_count=0,
        )

    aligned_shapes, mean_shape = generalized_procrustes(pre_shapes)
    for record, aligned in zip(processed_records, aligned_shapes):
        record.aligned_points = aligned

    matrix = aligned_shapes.reshape(len(aligned_shapes), -1).copy()
    matrix -= matrix.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = (singular_values ** 2) / max(len(processed_records) - 1, 1)
    explained = eigenvalues / eigenvalues.sum() if eigenvalues.size and eigenvalues.sum() > 0 else np.zeros(0)
    scores = u * singular_values if singular_values.size else np.zeros((len(processed_records), 0))
    default_pc_count = 0
    if explained.size:
        default_pc_count = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)

    factor_types, categorical_factors, continuous_factors = collect_available_factors(processed_records)

    return ShapeAnalysisResults(
        raw_dataset=dataset,
        records=processed_records,
        mean_shape=mean_shape,
        pca_scores=scores,
        explained_variance_ratio=explained,
        eigenvalues=eigenvalues,
        eigenvectors=vt,
        factor_types=factor_types,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        default_pc_count=default_pc_count,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_min = 1.0
    total = len(p_values)
    for rank, (index, value) in enumerate(reversed(indexed), start=1):
        adjusted_value = min(running_min, value * total / (total - rank + 1))
        adjusted[index] = adjusted_value
        running_min = adjusted_value
    return adjusted


def adjust_p_values(p_values: Sequence[float], method: str) -> List[float]:
    if not p_values:
        return []
    if method == "Bonferroni":
        return [min(value * len(p_values), 1.0) for value in p_values]
    if method == "FDR":
        return benjamini_hochberg(p_values)
    return list(p_values)


def one_way_manova(Y: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    labels_array = np.asarray(labels)
    unique_labels = [label for label in sorted(set(labels_array)) if np.sum(labels_array == label) > 0]
    n_samples, n_vars = Y.shape
    group_count = len(unique_labels)
    grand_mean = Y.mean(axis=0)
    hypothesis = np.zeros((n_vars, n_vars), dtype=float)
    error = np.zeros((n_vars, n_vars), dtype=float)

    for label in unique_labels:
        group = Y[labels_array == label]
        mean = group.mean(axis=0)
        delta = (mean - grand_mean).reshape(-1, 1)
        hypothesis += len(group) * (delta @ delta.T)
        centered = group - mean
        error += centered.T @ centered

    pillai = float(np.trace(hypothesis @ np.linalg.pinv(hypothesis + error)))
    q = group_count - 1
    s = min(n_vars, q)
    ve = n_samples - group_count
    m = (abs(n_vars - q) - 1.0) / 2.0
    n_term = (ve - n_vars - 1.0) / 2.0
    df1 = s * (2.0 * m + s + 1.0)
    df2 = s * (2.0 * n_term + s + 1.0)

    approx_f = float("nan")
    p_value = float("nan")
    if s > 0 and df1 > 0 and df2 > 0 and pillai < s:
        approx_f = ((2.0 * n_term + s + 1.0) * pillai) / ((2.0 * m + s + 1.0) * (s - pillai))
        p_value = float(1.0 - f.cdf(approx_f, df1, df2))

    return {
        "pillai": pillai,
        "approx_f": approx_f,
        "p_value": p_value,
        "df1": df1,
        "df2": df2,
        "groups": group_count,
        "n": n_samples,
    }


def permanova_euclidean(Y: np.ndarray, labels: Sequence[str], permutations: int = PERMANOVA_PERMUTATIONS) -> Dict[str, Any]:
    labels_array = np.asarray(labels)
    unique_labels = [label for label in sorted(set(labels_array)) if np.sum(labels_array == label) > 0]
    n_samples = len(labels_array)
    g_count = len(unique_labels)

    if g_count < 2 or n_samples <= g_count:
        return {"pseudo_f": float("nan"), "r_squared": float("nan"), "p_value": float("nan")}

    grand_mean = Y.mean(axis=0)
    total = float(np.sum((Y - grand_mean) ** 2))
    if total <= 0:
        return {"pseudo_f": 0.0, "r_squared": 0.0, "p_value": 1.0}

    # Precompute indices corresponding to each group to bypass expensive string matching inside permutation loop
    label_to_indices = {label: np.where(labels_array == label)[0] for label in unique_labels}

    def compute_between(shuffled_indices: np.ndarray) -> float:
        between = 0.0
        for label, idxs in label_to_indices.items():
            group_Y = Y[shuffled_indices[idxs]]
            if len(group_Y) == 0:
                continue
            mean = group_Y.mean(axis=0)
            between += len(group_Y) * np.sum((mean - grand_mean) ** 2)
        return float(between)

    # Calculate observed statistics
    observed_between = compute_between(np.arange(n_samples))
    observed_within = total - observed_between
    if observed_within <= 0:
        return {"pseudo_f": float("nan"), "r_squared": float("nan"), "p_value": float("nan")}

    observed_f = (observed_between / (g_count - 1)) / (observed_within / (n_samples - g_count))
    observed_r2 = observed_between / total

    # Perform permutation test of indices (mathematically equivalent and significantly faster than string shuffling)
    more_extreme = 1
    rng = np.random.default_rng(42)
    indices = np.arange(n_samples)
    for _ in range(permutations):
        rng.shuffle(indices)
        perm_between = compute_between(indices)
        if perm_between >= observed_between - 1e-10:
            more_extreme += 1

    p_value = more_extreme / float(permutations + 1)
    return {"pseudo_f": observed_f, "r_squared": observed_r2, "p_value": p_value}


def run_shape_statistics(results: ShapeAnalysisResults, pc_count: int, correction_method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if results.pca_scores.size == 0 or pc_count <= 0:
        return rows

    max_available_pcs = results.pca_scores.shape[1]
    requested_pcs = max(1, min(pc_count, max_available_pcs))

    for factor in results.categorical_factors:
        labels = [record.factor_values.get(factor) for record in results.records]
        mask = np.array([value is not None for value in labels], dtype=bool)
        if mask.sum() < 6:
            continue
        label_values = np.asarray([format_value(labels[index]) for index in np.where(mask)[0]])
        group_sizes = {group: int(np.sum(label_values == group)) for group in sorted(set(label_values))}
        if len(group_sizes) < 2:
            continue
        effective_pcs = min(requested_pcs, int(mask.sum()) - len(group_sizes) - 1, max_available_pcs)
        if effective_pcs < 1:
            continue
        Y = results.pca_scores[mask, :effective_pcs]

        manova = one_way_manova(Y, label_values)
        rows.append(
            {
                "Factor": factor,
                "Factor Type": "Categorical",
                "Test": "MANOVA",
                "Components": f"PC1-PC{effective_pcs}",
                "N": int(mask.sum()),
                "Groups": len(group_sizes),
                "Statistic": "Approx F",
                "Statistic Value": manova["approx_f"],
                "Effect Size": "Pillai's Trace",
                "Effect Value": manova["pillai"],
                "P Value": manova["p_value"],
            }
        )

        permanova = permanova_euclidean(Y, label_values)
        rows.append(
            {
                "Factor": factor,
                "Factor Type": "Categorical",
                "Test": "PERMANOVA",
                "Components": f"PC1-PC{effective_pcs}",
                "N": int(mask.sum()),
                "Groups": len(group_sizes),
                "Statistic": "Pseudo-F",
                "Statistic Value": permanova["pseudo_f"],
                "Effect Size": "R^2",
                "Effect Value": permanova["r_squared"],
                "P Value": permanova["p_value"],
            }
        )

    for factor in results.continuous_factors:
        numeric = np.array([safe_float(record.factor_values.get(factor)) for record in results.records], dtype=float)
        mask = ~np.isnan(numeric)
        if mask.sum() < 6:
            continue
        effective_pcs = min(requested_pcs, max_available_pcs)
        Y = results.pca_scores[mask, :effective_pcs]
        X = numeric[mask]
        for component_index in range(effective_pcs):
            regression = linregress(X, Y[:, component_index])
            rows.append(
                {
                    "Factor": factor,
                    "Factor Type": "Continuous",
                    "Test": "Linear Regression",
                    "Components": f"PC{component_index + 1}",
                    "N": int(mask.sum()),
                    "Groups": "",
                    "Statistic": "Slope",
                    "Statistic Value": regression.slope,
                    "Effect Size": "R^2",
                    "Effect Value": regression.rvalue ** 2,
                    "P Value": regression.pvalue,
                }
            )

    valid_indices = [index for index, row in enumerate(rows) if not math.isnan(float(row["P Value"]))]
    valid_p_values = [float(rows[index]["P Value"]) for index in valid_indices]
    adjusted = adjust_p_values(valid_p_values, correction_method)
    for row in rows:
        row["Adjusted P"] = float("nan")
        row["Correction"] = correction_method
    for index, adjusted_value in zip(valid_indices, adjusted):
        rows[index]["Adjusted P"] = adjusted_value
    return rows


class ShapeAnalysisTab(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        bg_color: str,
        fg_color: str,
        accent_color: str,
        button_bg: str,
        button_active: str,
        log_callback: Optional[Any] = None,
        status_callback: Optional[Any] = None,
    ) -> None:
        super().__init__(parent, bg=bg_color)
        self.pack(fill=tk.BOTH, expand=True)

        self.bg_color = bg_color
        self.fg_color = fg_color
        self.accent_color = accent_color
        self.button_bg = button_bg
        self.button_active = button_active
        self.card_bg = "#252526"
        self.border_color = "#3E3E3E"
        self.muted = "#A8A8A8"
        self.grid_line = "#4D4D4D"
        self.log_callback = log_callback
        self.status_callback = status_callback

        self.project_root = os.path.abspath(os.path.dirname(__file__))
        
        default_emory = r"E:\Emory"
        if not os.path.exists(default_emory):
            default_emory = os.path.normpath(os.path.join(self.project_root, "data"))
        self.emory_root_var = tk.StringVar(value=default_emory)

        self.manual_overrides: Dict[str, ManualOverride] = {}
        self.raw_dataset: Optional[RawDataset] = None
        self.results: Optional[ShapeAnalysisResults] = None
        self.stats_rows: List[Dict[str, Any]] = []
        self._analysis_generation = 0
        self._statistics_generation = 0
        self._scatter_payload: Dict[str, Any] = {}
        self._selected_uid: Optional[str] = None
        self._ui_task_queue: "queue.SimpleQueue[Callable[[], None]]" = queue.SimpleQueue()
        self._ui_pump_job: Optional[str] = None

        self.status_var = tk.StringVar(value="Ready")
        self.summary_var = tk.StringVar(value="Scan pending.")
        self.stats_status_var = tk.StringVar(value="Statistics have not been run yet.")
        self.preview_var = tk.StringVar(value="Select a subject in the audit table to inspect overrides.")

        self.color_factor_var = tk.StringVar(value="")
        self.overlay_mode_var = tk.StringVar(value="Convex Hulls")
        self.filter_factor_var = tk.StringVar(value=NO_FILTER_LABEL)
        self.x_pc_var = tk.StringVar(value="PC1")
        self.y_pc_var = tk.StringVar(value="PC2")
        self.override_rotation_var = tk.StringVar(value="0")
        self.override_mirror_var = tk.StringVar(value="Auto")
        self.stats_pc_count_var = tk.StringVar(value="1")
        self.stats_correction_var = tk.StringVar(value="FDR")

        self.dimension_vars = {
            metric: tk.BooleanVar(value=(metric in DEFAULT_DIMENSIONS)) for metric in METRIC_SPECS
        }

        self._configure_styles()
        self._build_scroll_shell()
        self._build_ui()
        self._schedule_ui_pump()

    def destroy(self) -> None:
        if self._ui_pump_job is not None:
            try:
                self.after_cancel(self._ui_pump_job)
            except Exception:
                pass
            self._ui_pump_job = None
        super().destroy()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.configure(
            "Shape.Treeview",
            background="#202020",
            fieldbackground="#202020",
            foreground=self.fg_color,
            rowheight=24,
            bordercolor=self.border_color,
        )
        style.map("Shape.Treeview", background=[("selected", self.accent_color)], foreground=[("selected", "white")])
        style.configure("Shape.Treeview.Heading", background="#2B2B2B", foreground="white")

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        if self.status_callback:
            self.status_callback(message)

    def _enqueue_ui_task(self, callback: Callable[[], None]) -> None:
        self._ui_task_queue.put(callback)

    def _schedule_ui_pump(self) -> None:
        self._ui_pump_job = self.after(50, self._drain_ui_tasks)

    def _drain_ui_tasks(self) -> None:
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
            if self.winfo_exists():
                self._schedule_ui_pump()
        except tk.TclError:
            return

    def _build_scroll_shell(self) -> None:
        self.scroll_canvas = tk.Canvas(
            self,
            bg=self.bg_color,
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content_frame = tk.Frame(self.scroll_canvas, bg=self.bg_color)
        self._scroll_window = self.scroll_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.content_frame.bind("<Configure>", self._on_scroll_content_configure)
        self.scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_scroll_content_configure(self, _event: Any) -> None:
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_scroll_canvas_configure(self, event: Any) -> None:
        self.scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

    def _widget_is_descendant(self, widget: Any) -> bool:
        current = widget
        while current is not None:
            if current == self:
                return True
            parent_name = getattr(current, "winfo_parent", lambda: "")()
            if not parent_name:
                break
            try:
                current = current.nametowidget(parent_name)
            except Exception:
                break
        return False

    def _on_mousewheel(self, event: Any) -> None:
        if not self.winfo_ismapped():
            return
        if not self._widget_is_descendant(event.widget):
            return
        if getattr(event.widget, "winfo_class", lambda: "")() in {"Treeview", "Listbox", "Text"}:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        steps = int(-delta / 120)
        if steps == 0:
            steps = -1 if delta > 0 else 1
        self.scroll_canvas.yview_scroll(steps, "units")

    def _card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(
            parent,
            bg=self.card_bg,
            highlightbackground=self.border_color,
            highlightthickness=1,
            padx=10,
            pady=10,
        )
        tk.Label(
            outer,
            text=title,
            bg=self.card_bg,
            fg=self.accent_color,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))
        return outer

    def _style_axis(self, ax: Any, title: str = "", xlabel: str = "", ylabel: str = "", grid: bool = True) -> None:
        ax.set_facecolor(self.card_bg)
        ax.figure.set_facecolor(self.card_bg)
        ax.tick_params(colors=self.fg_color, labelsize=8)
        ax.xaxis.label.set_color(self.fg_color)
        ax.yaxis.label.set_color(self.fg_color)
        ax.title.set_color(self.accent_color)
        if title:
            ax.set_title(title, fontsize=10, color=self.accent_color)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        for spine in ax.spines.values():
            spine.set_color(self.border_color)
        if grid:
            ax.grid(True, color=self.grid_line, alpha=0.22, linewidth=0.7)
        else:
            ax.grid(False)

    def _build_ui(self) -> None:
        header = tk.Frame(self.content_frame, bg=self.bg_color)
        header.pack(fill=tk.X, padx=16, pady=(14, 8))

        tk.Label(
            header,
            text="Shape Analysis",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            header,
            text=(
                "Exploratory 2D contour-based statistical shape analysis derived from data/labels/seg. "
                "This workspace does not contain true 3D MRI/CT meshes or calibrated spacing, so volume/surface "
                "metrics are not available."
            ),
            bg=self.bg_color,
            fg=self.muted,
            wraplength=1180,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(4, 8))

        controls = tk.Frame(header, bg=self.bg_color)
        controls.pack(fill=tk.X)
        tk.Button(
            controls,
            text="Refresh / Rescan",
            command=lambda: self.refresh_analysis(rescan=True),
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
        ).pack(side=tk.LEFT)
        tk.Button(
            controls,
            text="Export Raw CSV",
            command=self._export_raw_shape_csv,
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(
            controls,
            textvariable=self.status_var,
            bg=self.bg_color,
            fg=self.accent_color,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(12, 0))

        # ── Folder Paths Configuration Card ───────────────────────
        cfg_card = self._card(self.content_frame, "Folder Paths Configuration")
        cfg_card.pack(fill=tk.X, padx=16, pady=(0, 10))

        grid_frame = tk.Frame(cfg_card, bg=self.card_bg)
        grid_frame.pack(fill=tk.X)

        tk.Label(
            grid_frame,
            text="Emory Metadata Root:",
            bg=self.card_bg,
            fg=self.fg_color,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(5, 5))

        ent = tk.Entry(
            grid_frame,
            textvariable=self.emory_root_var,
            width=60,
            bg="#3C3C3C",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT,
        )
        ent.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ent.bind("<Return>", lambda _event: self.refresh_analysis(rescan=True))

        btn = tk.Button(
            grid_frame,
            text="Browse...",
            font=("Segoe UI", 8),
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
            command=self._browse_emory_root,
            padx=12,
            cursor="hand2",
        )
        btn.pack(side=tk.LEFT, padx=(0, 10))

        summary_card = self._card(self.content_frame, "Cohort Summary")
        summary_card.pack(fill=tk.X, padx=16, pady=(0, 10))
        tk.Label(
            summary_card,
            textvariable=self.summary_var,
            bg=self.card_bg,
            fg=self.fg_color,
            justify=tk.LEFT,
            anchor=tk.W,
            wraplength=1200,
            font=("Consolas", 9),
        ).pack(fill=tk.X)

        self.inner_notebook = ttk.Notebook(self.content_frame)
        self.inner_notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 14))

        self.overview_tab = tk.Frame(self.inner_notebook, bg=self.bg_color)
        self.shape_space_tab = tk.Frame(self.inner_notebook, bg=self.bg_color)
        self.statistics_tab = tk.Frame(self.inner_notebook, bg=self.bg_color)
        self.inner_notebook.add(self.overview_tab, text="Overview")
        self.inner_notebook.add(self.shape_space_tab, text="Shape Space")
        self.inner_notebook.add(self.statistics_tab, text="Statistics")

        self._build_overview_tab()
        self._build_shape_space_tab()
        self._build_statistics_tab()

    def _build_overview_tab(self) -> None:
        plots_row = tk.Frame(self.overview_tab, bg=self.bg_color)
        plots_row.pack(fill=tk.BOTH, expand=True)

        alignment_card = self._card(plots_row, "Alignment Quality")
        alignment_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=(0, 10))
        self.alignment_figure = Figure(figsize=(6.8, 4.4), dpi=100, facecolor=self.card_bg)
        self.alignment_canvas = FigureCanvasTkAgg(self.alignment_figure, master=alignment_card)
        self.alignment_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        dimension_card = self._card(plots_row, "Dimension Distributions")
        dimension_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 10))
        toggle_frame = tk.Frame(dimension_card, bg=self.card_bg)
        toggle_frame.pack(fill=tk.X, pady=(0, 8))
        for metric, label in METRIC_SPECS.items():
            tk.Checkbutton(
                toggle_frame,
                text=label,
                variable=self.dimension_vars[metric],
                command=self._render_dimension_figure,
                bg=self.card_bg,
                fg=self.fg_color,
                selectcolor="#333333",
                activebackground=self.card_bg,
                activeforeground=self.fg_color,
                font=("Segoe UI", 8),
            ).pack(side=tk.LEFT, padx=(0, 6))
        self.dimension_figure = Figure(figsize=(6.8, 4.4), dpi=100, facecolor=self.card_bg)
        self.dimension_canvas = FigureCanvasTkAgg(self.dimension_figure, master=dimension_card)
        self.dimension_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        lower_row = tk.Frame(self.overview_tab, bg=self.bg_color)
        lower_row.pack(fill=tk.BOTH, expand=True)

        audit_card = self._card(lower_row, "Audit Table")
        audit_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        columns = ("patient", "uid", "side", "mirror", "auto", "extra", "warnings")
        self.audit_tree = ttk.Treeview(audit_card, columns=columns, show="headings", style="Shape.Treeview")
        headings = {
            "patient": "Patient",
            "uid": "Study UID",
            "side": "Laterality",
            "mirror": "Mirrored",
            "auto": "Auto Rot",
            "extra": "Extra Rot",
            "warnings": "Warnings",
        }
        widths = {"patient": 90, "uid": 290, "side": 80, "mirror": 70, "auto": 70, "extra": 70, "warnings": 260}
        for column in columns:
            self.audit_tree.heading(column, text=headings[column])
            self.audit_tree.column(column, width=widths[column], stretch=(column in {"uid", "warnings"}))
        self.audit_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        audit_scroll = ttk.Scrollbar(audit_card, orient=tk.VERTICAL, command=self.audit_tree.yview)
        audit_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.audit_tree.configure(yscrollcommand=audit_scroll.set)
        self.audit_tree.bind("<<TreeviewSelect>>", self._on_audit_select)

        preview_card = self._card(lower_row, "Subject Preview / Manual Override")
        preview_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        tk.Label(
            preview_card,
            textvariable=self.preview_var,
            bg=self.card_bg,
            fg=self.fg_color,
            justify=tk.LEFT,
            anchor=tk.W,
            wraplength=430,
            font=("Consolas", 8),
        ).pack(fill=tk.X)

        override_row = tk.Frame(preview_card, bg=self.card_bg)
        override_row.pack(fill=tk.X, pady=(8, 8))
        tk.Label(override_row, text="Extra Rotation:", bg=self.card_bg, fg=self.fg_color).pack(side=tk.LEFT)
        self.override_rotation_cb = ttk.Combobox(
            override_row,
            textvariable=self.override_rotation_var,
            values=["0", "90", "180", "270"],
            width=6,
            state="readonly",
        )
        self.override_rotation_cb.pack(side=tk.LEFT, padx=(6, 12))
        tk.Label(override_row, text="Mirror:", bg=self.card_bg, fg=self.fg_color).pack(side=tk.LEFT)
        self.override_mirror_cb = ttk.Combobox(
            override_row,
            textvariable=self.override_mirror_var,
            values=["Auto", "Force Mirror", "Force No Mirror"],
            width=14,
            state="readonly",
        )
        self.override_mirror_cb.pack(side=tk.LEFT, padx=(6, 12))
        tk.Button(
            override_row,
            text="Apply",
            command=self._apply_override,
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            override_row,
            text="Reset",
            command=self._reset_override,
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
        ).pack(side=tk.LEFT)

        self.preview_figure = Figure(figsize=(5.4, 4.4), dpi=100, facecolor=self.card_bg)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_card)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_shape_space_tab(self) -> None:
        left = self._card(self.shape_space_tab, "Clinical Factor Controls")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        control_column = tk.Frame(left, bg=self.card_bg)
        control_column.pack(fill=tk.Y, expand=True)

        tk.Label(control_column, text="Colour By", bg=self.card_bg, fg=self.fg_color).pack(anchor=tk.W)
        self.color_factor_cb = ttk.Combobox(control_column, textvariable=self.color_factor_var, state="readonly", width=28)
        self.color_factor_cb.pack(fill=tk.X, pady=(0, 8))
        self.color_factor_cb.bind("<<ComboboxSelected>>", lambda _event: self._render_shape_space())

        tk.Label(control_column, text="Group Overlay", bg=self.card_bg, fg=self.fg_color).pack(anchor=tk.W)
        self.overlay_mode_cb = ttk.Combobox(
            control_column,
            textvariable=self.overlay_mode_var,
            values=["Neither", "Convex Hulls", "Density Contours"],
            state="readonly",
            width=28,
        )
        self.overlay_mode_cb.pack(fill=tk.X, pady=(0, 8))
        self.overlay_mode_cb.bind("<<ComboboxSelected>>", lambda _event: self._render_shape_space())

        tk.Label(control_column, text="Filter Variable", bg=self.card_bg, fg=self.fg_color).pack(anchor=tk.W)
        self.filter_factor_cb = ttk.Combobox(
            control_column,
            textvariable=self.filter_factor_var,
            values=[NO_FILTER_LABEL],
            state="readonly",
            width=28,
        )
        self.filter_factor_cb.pack(fill=tk.X, pady=(0, 8))
        self.filter_factor_cb.bind("<<ComboboxSelected>>", lambda _event: self._refresh_filter_values())

        tk.Label(control_column, text="Visible Groups", bg=self.card_bg, fg=self.fg_color).pack(anchor=tk.W)
        self.filter_values_listbox = tk.Listbox(
            control_column,
            bg="#1F1F1F",
            fg=self.fg_color,
            selectbackground=self.accent_color,
            selectforeground="white",
            height=10,
            exportselection=False,
        )
        self.filter_values_listbox.pack(fill=tk.X, pady=(0, 8))
        self.filter_values_listbox.bind("<<ListboxSelect>>", lambda _event: self._render_shape_space())

        axis_row = tk.Frame(control_column, bg=self.card_bg)
        axis_row.pack(fill=tk.X, pady=(4, 8))
        tk.Label(axis_row, text="X Axis", bg=self.card_bg, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        tk.Label(axis_row, text="Y Axis", bg=self.card_bg, fg=self.fg_color).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.x_pc_cb = ttk.Combobox(axis_row, textvariable=self.x_pc_var, state="readonly", width=10)
        self.y_pc_cb = ttk.Combobox(axis_row, textvariable=self.y_pc_var, state="readonly", width=10)
        self.x_pc_cb.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        self.y_pc_cb.grid(row=1, column=1, padx=(8, 0), pady=(8, 0), sticky="ew")
        axis_row.columnconfigure(1, weight=1)
        self.x_pc_cb.bind("<<ComboboxSelected>>", lambda _event: self._render_shape_space())
        self.y_pc_cb.bind("<<ComboboxSelected>>", lambda _event: self._render_shape_space())

        tk.Button(
            control_column,
            text="Reset Filters",
            command=self._reset_filters,
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
        ).pack(anchor=tk.W, pady=(6, 8))

        tk.Label(
            control_column,
            text=(
                "Hover the scatter to inspect patient ID, laterality, corrections, and the currently selected factor. "
                "Missing factor values remain visible in grey."
            ),
            bg=self.card_bg,
            fg=self.muted,
            justify=tk.LEFT,
            wraplength=260,
            font=("Segoe UI", 8),
        ).pack(fill=tk.X)

        right = tk.Frame(self.shape_space_tab, bg=self.bg_color)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scatter_card = self._card(right, "PCA Shape Space")
        scatter_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.scatter_figure = Figure(figsize=(9.2, 5.4), dpi=100, facecolor=self.card_bg)
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_figure, master=scatter_card)
        self.scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.scatter_canvas.mpl_connect("motion_notify_event", self._on_scatter_hover)

        lower = tk.Frame(right, bg=self.bg_color)
        lower.pack(fill=tk.BOTH, expand=True)

        scree_card = self._card(lower, "Variance Explained")
        scree_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.scree_figure = Figure(figsize=(4.5, 3.7), dpi=100, facecolor=self.card_bg)
        self.scree_canvas = FigureCanvasTkAgg(self.scree_figure, master=scree_card)
        self.scree_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        deformation_card = self._card(lower, "Mean Shape +/- Deformation")
        deformation_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.deformation_figure = Figure(figsize=(5.4, 3.7), dpi=100, facecolor=self.card_bg)
        self.deformation_canvas = FigureCanvasTkAgg(self.deformation_figure, master=deformation_card)
        self.deformation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_statistics_tab(self) -> None:
        top = self._card(self.statistics_tab, "Exploratory Statistics")
        top.pack(fill=tk.X, pady=(0, 10))

        controls = tk.Frame(top, bg=self.card_bg)
        controls.pack(fill=tk.X)
        tk.Label(controls, text="Top PCs", bg=self.card_bg, fg=self.fg_color).pack(side=tk.LEFT)
        self.stats_pc_spinbox = tk.Spinbox(
            controls,
            from_=1,
            to=1,
            textvariable=self.stats_pc_count_var,
            width=6,
            bg="#1F1F1F",
            fg=self.fg_color,
            insertbackground=self.fg_color,
            buttonbackground="#2A2A2A",
        )
        self.stats_pc_spinbox.pack(side=tk.LEFT, padx=(6, 14))
        tk.Label(controls, text="Multiple Comparison Correction", bg=self.card_bg, fg=self.fg_color).pack(side=tk.LEFT)
        self.stats_correction_cb = ttk.Combobox(
            controls,
            textvariable=self.stats_correction_var,
            values=["None", "Bonferroni", "FDR"],
            state="readonly",
            width=14,
        )
        self.stats_correction_cb.pack(side=tk.LEFT, padx=(6, 14))
        tk.Button(
            controls,
            text="Run / Refresh Statistics",
            command=self._run_statistics_async,
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
        ).pack(side=tk.LEFT)
        tk.Button(
            controls,
            text="Export CSV",
            command=self._export_statistics_csv,
            bg=self.button_bg,
            fg="white",
            activebackground=self.button_active,
            activeforeground="white",
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(
            top,
            text=(
                "These analyses are exploratory and are computed on a segmentation contour dataset. "
                "They should be interpreted cautiously and are not a substitute for study-specific inferential design."
            ),
            bg=self.card_bg,
            fg="#F0C674",
            wraplength=1200,
            justify=tk.LEFT,
            font=("Segoe UI", 9, "italic"),
        ).pack(anchor=tk.W, pady=(8, 4))
        tk.Label(top, textvariable=self.stats_status_var, bg=self.card_bg, fg=self.muted).pack(anchor=tk.W)

        table_card = self._card(self.statistics_tab, "Results")
        table_card.pack(fill=tk.BOTH, expand=True)
        columns = ("factor", "type", "test", "components", "n", "stat", "effect", "p", "adj")
        self.stats_tree = ttk.Treeview(table_card, columns=columns, show="headings", style="Shape.Treeview")
        headings = {
            "factor": "Factor",
            "type": "Type",
            "test": "Test",
            "components": "Components",
            "n": "N",
            "stat": "Statistic",
            "effect": "Effect Size",
            "p": "P Value",
            "adj": "Adjusted P",
        }
        widths = {"factor": 170, "type": 90, "test": 120, "components": 120, "n": 60, "stat": 130, "effect": 130, "p": 90, "adj": 90}
        for column in columns:
            self.stats_tree.heading(column, text=headings[column])
            self.stats_tree.column(column, width=widths[column], stretch=(column == "factor"))
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll = ttk.Scrollbar(table_card, orient=tk.VERTICAL, command=self.stats_tree.yview)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)

    def _browse_emory_root(self) -> None:
        initial = self.emory_root_var.get()
        if not os.path.exists(initial):
            initial = os.getcwd()
        d = filedialog.askdirectory(parent=self, title="Select Emory Clinical Metadata Root Directory", initialdir=initial)
        if d:
            self.emory_root_var.set(os.path.normpath(d))
            self.refresh_analysis(rescan=True)

    def refresh_analysis(self, *, rescan: bool) -> None:
        self._analysis_generation += 1
        generation = self._analysis_generation
        self._set_status("Loading shape-analysis cohort...")
        self._log("Shape analysis: loading segmentation contours and Emory metadata")

        def worker() -> None:
            def worker_log(message: str) -> None:
                self._enqueue_ui_task(lambda message=message: self._log(message))

            try:
                dataset = self.raw_dataset
                if rescan or dataset is None:
                    dataset = load_raw_dataset(self.project_root, self.emory_root_var.get(), worker_log)
                results = process_dataset(dataset, self.manual_overrides)
                self._enqueue_ui_task(
                    lambda dataset=dataset, results=results: self._on_analysis_complete(generation, dataset, results)
                )
            except Exception as exc:
                self._enqueue_ui_task(lambda exc=exc: self._on_analysis_error(generation, exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_analysis_complete(self, generation: int, dataset: RawDataset, results: ShapeAnalysisResults) -> None:
        if generation != self._analysis_generation:
            return
        self.raw_dataset = dataset
        self.results = results
        self._set_status("Shape-analysis cohort ready")

        # Friendly scan-completed explanatory popup informing the user of the PCA & metrics pipeline rendering sequence
        messagebox.showinfo(
            "Database Scan Complete",
            "Clinical database scan completed successfully!\n\n"
            "Click OK to render the shape workspace plots and prepare PCA results.\n"
            "This sequence will design PCA shape-spaces, compile demographics, and render metrics tables.",
            parent=self
        )

        self._update_summary()
        self._populate_audit_table()
        self._refresh_factor_controls()
        self._refresh_pc_controls()
        self._render_alignment_figure()
        self._render_dimension_figure()
        self._render_shape_space()
        self._render_scree_figure()
        self._render_deformation_figure()
        self._render_preview()
        self._run_statistics_async()

    def _on_analysis_error(self, generation: int, exc: Exception) -> None:
        if generation != self._analysis_generation:
            return
        self._set_status("Shape-analysis load failed")
        messagebox.showerror("Shape Analysis", str(exc))

    def _update_summary(self) -> None:
        if not self.results:
            self.summary_var.set("No shape-analysis results available.")
            return
        summary = self.results.raw_dataset.summary
        lines = [
            f"Contours loaded:           {summary['segmentation_count']}",
            f"Matched image metadata:    {summary['matched_image_metadata']}",
            f"Matched demographics:      {summary['matched_demographics']}",
            f"Matched exact pain score:  {summary['matched_pain']}",
            f"Missing image files:       {summary['missing_image_files']}",
            f"Emory image rows w/o seg:  {summary['unmatched_image_metadata_rows']}",
            f"Standard laterality:       {STANDARD_LATERALITY} knee convention",
            "Processing steps:          laterality harmonisation -> PCA-axis rotation -> generalized Procrustes",
        ]
        if summary["critical_warnings"]:
            lines.append(f"Critical warnings:         {len(summary['critical_warnings'])}")
        self.summary_var.set("\n".join(lines))

    def _populate_audit_table(self) -> None:
        self.audit_tree.delete(*self.audit_tree.get_children())
        if not self.results:
            return
        selected_uid = self._selected_uid
        for record in self.results.records:
            warnings = "; ".join(record.warnings) if record.warnings else "OK"
            self.audit_tree.insert(
                "",
                tk.END,
                iid=record.raw.uid,
                values=(
                    record.raw.empi or MISSING_VALUE_LABEL,
                    record.raw.uid,
                    format_value(record.factor_values.get("Original Laterality")),
                    "Yes" if record.mirror_applied else "No",
                    f"{record.auto_rotation_deg:.1f}°",
                    f"{record.extra_rotation_deg}°",
                    warnings,
                ),
            )
        if selected_uid and self.audit_tree.exists(selected_uid):
            self.audit_tree.selection_set(selected_uid)
            self.audit_tree.see(selected_uid)
        elif self.results.records:
            first_uid = self.results.records[0].raw.uid
            self.audit_tree.selection_set(first_uid)
            self._selected_uid = first_uid

    def _refresh_factor_controls(self) -> None:
        if not self.results:
            return
        color_options = self.results.categorical_factors + self.results.continuous_factors
        if color_options:
            default_color = self.color_factor_var.get()
            if default_color not in color_options:
                default_color = "Sex" if "Sex" in color_options else color_options[0]
            self.color_factor_cb.configure(values=color_options)
            self.color_factor_var.set(default_color)
        else:
            self.color_factor_cb.configure(values=[])
            self.color_factor_var.set("")
        filter_options = [NO_FILTER_LABEL] + self.results.categorical_factors
        default_filter = self.filter_factor_var.get()
        if default_filter not in filter_options:
            default_filter = "Original Laterality" if "Original Laterality" in filter_options else NO_FILTER_LABEL
        self.filter_factor_cb.configure(values=filter_options)
        self.filter_factor_var.set(default_filter)
        self._refresh_filter_values()

    def _refresh_filter_values(self) -> None:
        self.filter_values_listbox.delete(0, tk.END)
        if not self.results:
            return
        factor = self.filter_factor_var.get()
        if factor == NO_FILTER_LABEL:
            return
        values = sorted({format_value(record.factor_values.get(factor)) for record in self.results.records})
        for value in values:
            self.filter_values_listbox.insert(tk.END, value)
        self.filter_values_listbox.select_set(0, tk.END)
        self._render_shape_space()

    def _reset_filters(self) -> None:
        self.overlay_mode_var.set("Convex Hulls")
        self.filter_factor_var.set(NO_FILTER_LABEL)
        self._refresh_filter_values()
        self._render_shape_space()

    def _refresh_pc_controls(self) -> None:
        if not self.results or self.results.explained_variance_ratio.size == 0:
            return
        pc_names = [f"PC{index + 1}" for index in range(len(self.results.explained_variance_ratio))]
        self.x_pc_cb.configure(values=pc_names)
        self.y_pc_cb.configure(values=pc_names)
        if self.x_pc_var.get() not in pc_names:
            self.x_pc_var.set("PC1")
        if self.y_pc_var.get() not in pc_names:
            self.y_pc_var.set("PC2" if len(pc_names) > 1 else "PC1")
        default_pc_count = max(1, self.results.default_pc_count or 1)
        self.stats_pc_spinbox.config(to=len(pc_names))
        self.stats_pc_count_var.set(str(default_pc_count))

    def _selected_record(self) -> Optional[ProcessedShapeRecord]:
        if not self.results:
            return None
        selected = self.audit_tree.selection()
        if not selected:
            return None
        uid = selected[0]
        for record in self.results.records:
            if record.raw.uid == uid:
                return record
        return None

    def _on_audit_select(self, _event: Any) -> None:
        record = self._selected_record()
        if not record:
            return
        self._selected_uid = record.raw.uid
        override = self.manual_overrides.get(record.raw.uid, ManualOverride())
        self.override_rotation_var.set(str(override.extra_rotation_deg))
        if override.force_mirror is None:
            self.override_mirror_var.set("Auto")
        else:
            self.override_mirror_var.set("Force Mirror" if override.force_mirror else "Force No Mirror")
        self._render_preview()

    def _apply_override(self) -> None:
        record = self._selected_record()
        if not record:
            return
        extra_rotation = int(self.override_rotation_var.get())
        mirror_choice = self.override_mirror_var.get()
        if mirror_choice == "Force Mirror":
            force_mirror = True
        elif mirror_choice == "Force No Mirror":
            force_mirror = False
        else:
            force_mirror = None
        override = ManualOverride(extra_rotation_deg=extra_rotation, force_mirror=force_mirror)
        if override.extra_rotation_deg == 0 and override.force_mirror is None:
            self.manual_overrides.pop(record.raw.uid, None)
        else:
            self.manual_overrides[record.raw.uid] = override
        self._set_status(f"Recomputing alignment for {record.raw.uid}")
        self.refresh_analysis(rescan=False)

    def _reset_override(self) -> None:
        record = self._selected_record()
        if not record:
            return
        self.manual_overrides.pop(record.raw.uid, None)
        self.override_rotation_var.set("0")
        self.override_mirror_var.set("Auto")
        self.refresh_analysis(rescan=False)

    def _render_alignment_figure(self) -> None:
        figure = self.alignment_figure
        figure.clear()
        if not self.results or not self.results.records:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="Alignment Quality", grid=False)
            axis.text(0.5, 0.5, "No contours available.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.alignment_canvas.draw_idle()
            return
        axes = figure.subplots(1, 2)
        before_axis, after_axis = axes
        for axis in axes:
            self._style_axis(axis)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("X")
            axis.set_ylabel("Y")
        before_axis.set_title("Before Procrustes")
        after_axis.set_title("After Procrustes")

        # Migrate visualization to high-performance LineCollection arrays to prevent UI thread freezes
        from matplotlib.collections import LineCollection
        before_segments = [close_points(record.pre_procrustes_points) for record in self.results.records]
        after_segments = [close_points(record.aligned_points) for record in self.results.records]

        before_coll = LineCollection(before_segments, color="#78A7FF", alpha=0.12, linewidths=0.8)
        after_coll = LineCollection(after_segments, color="#78A7FF", alpha=0.12, linewidths=0.8)

        before_axis.add_collection(before_coll)
        after_axis.add_collection(after_coll)

        # Set manual limits because add_collection does not autoscale automatically
        if before_segments:
            all_before = np.concatenate(before_segments)
            bx_min, by_min = all_before.min(axis=0)
            bx_max, by_max = all_before.max(axis=0)
            bx_range = max(bx_max - bx_min, 1e-5)
            by_range = max(by_max - by_min, 1e-5)
            before_axis.set_xlim(bx_min - 0.05 * bx_range, bx_max + 0.05 * bx_range)
            before_axis.set_ylim(by_min - 0.05 * by_range, by_max + 0.05 * by_range)

            all_after = np.concatenate(after_segments)
            ax_min, ay_min = all_after.min(axis=0)
            ax_max, ay_max = all_after.max(axis=0)
            ax_range = max(ax_max - ax_min, 1e-5)
            ay_range = max(ay_max - ay_min, 1e-5)
            after_axis.set_xlim(ax_min - 0.05 * ax_range, ax_max + 0.05 * ax_range)
            after_axis.set_ylim(ay_min - 0.05 * ay_range, ay_max + 0.05 * ay_range)

        mean_shape = close_points(self.results.mean_shape)
        after_axis.plot(mean_shape[:, 0], mean_shape[:, 1], color="#F5B041", linewidth=2.2, label="Mean Shape")
        after_axis.legend(facecolor=self.card_bg, edgecolor=self.border_color, labelcolor=self.fg_color, fontsize=8)
        figure.tight_layout(pad=1.0)
        self.alignment_canvas.draw_idle()

    def _render_dimension_figure(self) -> None:
        figure = self.dimension_figure
        figure.clear()
        if not self.results or not self.results.records:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="Dimension Distributions", grid=False)
            axis.text(0.5, 0.5, "No metrics available.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.dimension_canvas.draw_idle()
            return
        selected_metrics = [metric for metric, variable in self.dimension_vars.items() if variable.get()]
        if not selected_metrics:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="Dimension Distributions", grid=False)
            axis.text(0.5, 0.5, "Select at least one metric.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.dimension_canvas.draw_idle()
            return
        columns = 2
        rows = int(math.ceil(len(selected_metrics) / columns))
        axes = figure.subplots(rows, columns, squeeze=False)
        for axis in axes.flatten():
            axis.set_visible(False)
        for index, metric in enumerate(selected_metrics):
            axis = axes[index // columns][index % columns]
            axis.set_visible(True)
            self._style_axis(axis, title=METRIC_SPECS[metric], xlabel="Value", ylabel="Count")
            values = [record.metrics[metric] for record in self.results.records if not math.isnan(record.metrics[metric])]
            if not values:
                axis.text(0.5, 0.5, "No data", transform=axis.transAxes, color=self.fg_color, ha="center")
                continue
            bins = min(20, max(6, int(math.sqrt(len(values)))))
            axis.hist(values, bins=bins, color="#4FC3F7", alpha=0.85, edgecolor="#111111")
            axis.axvline(float(np.mean(values)), color="#F5B041", linewidth=1.4, linestyle="--")
            axis.axvline(float(np.median(values)), color="#7ED957", linewidth=1.4, linestyle=":")
        figure.tight_layout(pad=1.0)
        self.dimension_canvas.draw_idle()

    def _visible_indices(self) -> List[int]:
        if not self.results:
            return []
        factor = self.filter_factor_var.get()
        if factor == NO_FILTER_LABEL:
            return list(range(len(self.results.records)))
        selected_indices = self.filter_values_listbox.curselection()
        if not selected_indices:
            return list(range(len(self.results.records)))
        selected_values = {self.filter_values_listbox.get(index) for index in selected_indices}
        visible = []
        for index, record in enumerate(self.results.records):
            if format_value(record.factor_values.get(factor)) in selected_values:
                visible.append(index)
        return visible

    def _parse_pc_index(self, value: str) -> int:
        if value.startswith("PC"):
            return max(int(value[2:]) - 1, 0)
        return 0

    def _build_scatter_colours(
        self, factor: str, indices: Sequence[int]
    ) -> Tuple[List[str], List[Line2D], Optional[ScalarMappable], Dict[int, str]]:
        if not self.results:
            return [], [], None, {}
        factor_type = self.results.factor_types.get(factor, "categorical")
        records = self.results.records
        tooltip_values: Dict[int, str] = {}
        if factor_type == "continuous":
            values = np.array([safe_float(records[index].factor_values.get(factor)) for index in indices], dtype=float)
            valid = ~np.isnan(values)
            colours = [MISSING_VALUE_LABEL] * len(indices)
            mappable: Optional[ScalarMappable] = None
            if valid.any():
                norm = Normalize(vmin=float(values[valid].min()), vmax=float(values[valid].max()))
                cmap = get_cmap("viridis")
                mappable = ScalarMappable(norm=norm, cmap=cmap)
                for idx, value in enumerate(values):
                    if math.isnan(value):
                        colours[idx] = "#8A8A8A"
                        tooltip_values[idx] = MISSING_VALUE_LABEL
                    else:
                        colours[idx] = to_hex(cmap(norm(value)))
                        tooltip_values[idx] = format_value(value)
            else:
                for idx in range(len(indices)):
                    colours[idx] = "#8A8A8A"
                    tooltip_values[idx] = MISSING_VALUE_LABEL
            return colours, [], mappable, tooltip_values

        categories = [format_value(records[index].factor_values.get(factor)) for index in indices]
        unique_categories = [value for value in sorted(set(categories)) if value != MISSING_VALUE_LABEL]
        palette = get_cmap("tab10")
        colour_map = {
            category: to_hex(palette(position % 10)) for position, category in enumerate(unique_categories)
        }
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", label=category, markerfacecolor=colour_map[category], markersize=7)
            for category in unique_categories
        ]
        colours = []
        for idx, category in enumerate(categories):
            tooltip_values[idx] = category
            colours.append(colour_map.get(category, "#8A8A8A"))
        if MISSING_VALUE_LABEL in categories:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="none", label=MISSING_VALUE_LABEL, markerfacecolor="#8A8A8A", markersize=7)
            )
        return colours, legend_handles, None, tooltip_values

    def _render_shape_space(self) -> None:
        figure = self.scatter_figure
        figure.clear()
        self._scatter_payload = {}
        if not self.results or self.results.pca_scores.size == 0:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="PCA Shape Space", grid=False)
            axis.text(0.5, 0.5, "PCA shape space is unavailable.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.scatter_canvas.draw_idle()
            return

        indices = self._visible_indices()
        if not indices:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="PCA Shape Space", grid=False)
            axis.text(0.5, 0.5, "No records match the current subgroup filter.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.scatter_canvas.draw_idle()
            return

        x_index = self._parse_pc_index(self.x_pc_var.get())
        y_index = self._parse_pc_index(self.y_pc_var.get())
        if y_index == x_index and self.results.pca_scores.shape[1] > 1:
            y_index = 1 if x_index == 0 else 0
            self.y_pc_var.set(f"PC{y_index + 1}")

        axis = figure.add_subplot(111)
        self._style_axis(
            axis,
            title="PCA Shape Space",
            xlabel=f"PC{x_index + 1} ({self.results.explained_variance_ratio[x_index] * 100:.1f}%)",
            ylabel=f"PC{y_index + 1} ({self.results.explained_variance_ratio[y_index] * 100:.1f}%)",
        )
        coordinates = self.results.pca_scores[np.asarray(indices), :][:, [x_index, y_index]]
        available_factors = self.results.categorical_factors + self.results.continuous_factors
        if not available_factors:
            axis.text(0.5, 0.5, "No joinable clinical or demographic factors are available.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.scatter_canvas.draw_idle()
            return
        factor = self.color_factor_var.get() if self.color_factor_var.get() else available_factors[0]
        colours, legend_handles, mappable, tooltip_values = self._build_scatter_colours(factor, indices)
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=colours,
            s=54,
            edgecolors="#111111",
            linewidths=0.45,
            alpha=0.9,
        )

        factor_type = self.results.factor_types.get(factor, "categorical")
        overlay_mode = self.overlay_mode_var.get()
        if factor_type == "categorical" and overlay_mode != "Neither":
            self._plot_group_overlay(axis, coordinates, factor, indices, colours, overlay_mode)
        elif factor_type == "continuous" and overlay_mode != "Neither":
            axis.text(
                0.01,
                0.99,
                "Group overlays apply to categorical factors only.",
                transform=axis.transAxes,
                color=self.muted,
                fontsize=8,
                ha="left",
                va="top",
            )

        if legend_handles:
            axis.legend(
                handles=legend_handles,
                facecolor=self.card_bg,
                edgecolor=self.border_color,
                labelcolor=self.fg_color,
                fontsize=8,
                loc="best",
            )
        if mappable is not None:
            colorbar = figure.colorbar(mappable, ax=axis, pad=0.02)
            colorbar.ax.tick_params(colors=self.fg_color, labelsize=8)
            colorbar.outline.set_edgecolor(self.border_color)
            colorbar.set_label(factor, color=self.fg_color)

        annotation = axis.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E1E1E", edgecolor=self.border_color),
            color="white",
            fontsize=8,
        )
        annotation.set_visible(False)

        self._scatter_payload = {
            "axis": axis,
            "coordinates": coordinates,
            "indices": indices,
            "annotation": annotation,
            "factor": factor,
            "tooltip_values": tooltip_values,
        }
        figure.tight_layout(pad=1.0)
        self.scatter_canvas.draw_idle()
        self._render_scree_figure()
        self._render_deformation_figure()

    def _plot_group_overlay(
        self,
        axis: Any,
        coordinates: np.ndarray,
        factor: str,
        indices: Sequence[int],
        colours: Sequence[str],
        overlay_mode: str,
    ) -> None:
        if not self.results:
            return
        group_to_points: Dict[str, List[np.ndarray]] = defaultdict(list)
        group_to_colour: Dict[str, str] = {}
        for local_index, record_index in enumerate(indices):
            label = format_value(self.results.records[record_index].factor_values.get(factor))
            if label == MISSING_VALUE_LABEL:
                continue
            group_to_points[label].append(coordinates[local_index])
            group_to_colour[label] = colours[local_index]

        x_pad = (coordinates[:, 0].max() - coordinates[:, 0].min()) * 0.12 + 1e-6
        y_pad = (coordinates[:, 1].max() - coordinates[:, 1].min()) * 0.12 + 1e-6
        x_grid = np.linspace(coordinates[:, 0].min() - x_pad, coordinates[:, 0].max() + x_pad, 100)
        y_grid = np.linspace(coordinates[:, 1].min() - y_pad, coordinates[:, 1].max() + y_pad, 100)
        xx, yy = np.meshgrid(x_grid, y_grid)

        for group, point_list in group_to_points.items():
            if len(point_list) < 3:
                continue
            points = np.asarray(point_list)
            colour = group_to_colour[group]
            if overlay_mode == "Convex Hulls":
                try:
                    hull = ConvexHull(points)
                    hull_points = close_points(points[hull.vertices])
                    axis.plot(hull_points[:, 0], hull_points[:, 1], color=colour, linewidth=1.25, alpha=0.9)
                    axis.fill(hull_points[:, 0], hull_points[:, 1], color=colour, alpha=0.06)
                except Exception:
                    continue
            elif overlay_mode == "Density Contours" and len(points) >= 5:
                try:
                    kde = gaussian_kde(points.T)
                    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    levels = np.quantile(zz, [0.65, 0.82])
                    axis.contour(xx, yy, zz, levels=levels, colors=[colour], linewidths=1.15, alpha=0.9)
                except Exception:
                    continue

    def _on_scatter_hover(self, event: Any) -> None:
        payload = self._scatter_payload
        if not payload:
            return
        annotation = payload["annotation"]
        axis = payload["axis"]
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                self.scatter_canvas.draw_idle()
            return
        coordinates = payload["coordinates"]
        distances = np.sqrt(np.sum((coordinates - np.array([event.xdata, event.ydata])) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        x_range = max(axis.get_xlim()[1] - axis.get_xlim()[0], 1e-6)
        y_range = max(axis.get_ylim()[1] - axis.get_ylim()[0], 1e-6)
        threshold = 0.03 * max(x_range, y_range)
        if distances[nearest_index] > threshold:
            if annotation.get_visible():
                annotation.set_visible(False)
                self.scatter_canvas.draw_idle()
            return
        record_index = payload["indices"][nearest_index]
        record = self.results.records[record_index] if self.results else None
        if not record:
            return
        factor = payload["factor"]
        factor_value = payload["tooltip_values"].get(nearest_index, MISSING_VALUE_LABEL)
        warnings = "; ".join(record.warnings) if record.warnings else "None"
        annotation.xy = tuple(coordinates[nearest_index])
        annotation.set_text(
            "\n".join(
                [
                    f"Patient: {record.raw.empi or MISSING_VALUE_LABEL}",
                    f"Study UID: {record.raw.uid}",
                    f"Laterality: {format_value(record.factor_values.get('Original Laterality'))}",
                    f"Corrections: {'Mirror ' if record.mirror_applied else ''}auto {record.auto_rotation_deg:.1f}°, extra {record.extra_rotation_deg}°",
                    f"{factor}: {factor_value}",
                    f"Warnings: {warnings}",
                ]
            )
        )
        annotation.set_visible(True)
        self.scatter_canvas.draw_idle()

    def _render_scree_figure(self) -> None:
        figure = self.scree_figure
        figure.clear()
        axis = figure.add_subplot(111)
        if not self.results or self.results.explained_variance_ratio.size == 0:
            self._style_axis(axis, title="Variance Explained", grid=False)
            axis.text(0.5, 0.5, "No PCA components available.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.scree_canvas.draw_idle()
            return
        explained = self.results.explained_variance_ratio[: min(12, len(self.results.explained_variance_ratio))]
        cumulative = np.cumsum(explained) * 100.0
        x = np.arange(1, len(explained) + 1)
        self._style_axis(axis, title="Variance Explained", xlabel="Principal Component", ylabel="% Variance")
        axis.bar(x, explained * 100.0, color="#4FC3F7", alpha=0.88)
        axis.plot(x, cumulative, color="#F5B041", marker="o", linewidth=1.6)
        axis.set_xticks(x)
        figure.tight_layout(pad=1.0)
        self.scree_canvas.draw_idle()

    def _render_deformation_figure(self) -> None:
        figure = self.deformation_figure
        figure.clear()
        if not self.results or self.results.eigenvectors.size == 0:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="Mean Shape +/- Deformation", grid=False)
            axis.text(0.5, 0.5, "No PCA deformation available.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.deformation_canvas.draw_idle()
            return
        axes = figure.subplots(1, 2)
        pc_indices = [self._parse_pc_index(self.x_pc_var.get()), self._parse_pc_index(self.y_pc_var.get())]
        for axis, pc_index in zip(axes, pc_indices):
            self._style_axis(
                axis,
                title=f"PC{pc_index + 1} ({self.results.explained_variance_ratio[pc_index] * 100:.1f}%)",
                grid=False,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
            mean_shape = self.results.mean_shape
            mode = self.results.eigenvectors[pc_index].reshape(-1, 2)
            spread = math.sqrt(float(self.results.eigenvalues[pc_index])) if pc_index < len(self.results.eigenvalues) else 0.0
            variants = [
                ("-2 SD", mean_shape - 2.0 * spread * mode, "#C62828"),
                ("-1 SD", mean_shape - 1.0 * spread * mode, "#FF7043"),
                ("Mean", mean_shape, "#4FC3F7"),
                ("+1 SD", mean_shape + 1.0 * spread * mode, "#66BB6A"),
                ("+2 SD", mean_shape + 2.0 * spread * mode, "#2E7D32"),
            ]
            for label, shape, colour in variants:
                closed = close_points(shape)
                axis.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=1.4 if label == "Mean" else 1.0, label=label)
            axis.legend(facecolor=self.card_bg, edgecolor=self.border_color, labelcolor=self.fg_color, fontsize=7)
        figure.tight_layout(pad=1.0)
        self.deformation_canvas.draw_idle()

    def _render_preview(self) -> None:
        figure = self.preview_figure
        figure.clear()
        record = self._selected_record()
        if not record:
            axis = figure.add_subplot(111)
            self._style_axis(axis, title="Subject Preview", grid=False)
            axis.text(0.5, 0.5, "Select a subject in the audit table.", transform=axis.transAxes, color=self.fg_color, ha="center")
            self.preview_canvas.draw_idle()
            return
        axes = figure.subplots(1, 2)
        raw_axis, corrected_axis = axes
        self._style_axis(raw_axis, title="Raw Outline", grid=False)
        self._style_axis(corrected_axis, title="Corrected Outline", grid=False)
        for axis in axes:
            axis.set_aspect("equal", adjustable="box")
            axis.axis("off")
        raw = close_points(record.raw.raw_points_px - record.raw.raw_points_px.mean(axis=0))
        corrected = close_points(record.corrected_points_px - record.corrected_points_px.mean(axis=0))
        raw_axis.fill(raw[:, 0], raw[:, 1], color="#546E7A", alpha=0.28)
        raw_axis.plot(raw[:, 0], raw[:, 1], color="#CFD8DC", linewidth=1.2)
        corrected_axis.fill(corrected[:, 0], corrected[:, 1], color="#4FC3F7", alpha=0.28)
        corrected_axis.plot(corrected[:, 0], corrected[:, 1], color="#E1F5FE", linewidth=1.2)
        self.preview_var.set(
            "\n".join(
                [
                    f"Patient: {record.raw.empi or MISSING_VALUE_LABEL}",
                    f"Study UID: {record.raw.uid}",
                    f"Study Date: {record.raw.study_date or MISSING_VALUE_LABEL}",
                    f"Original Laterality: {format_value(record.factor_values.get('Original Laterality'))}",
                    f"Mirror Applied: {'Yes' if record.mirror_applied else 'No'}",
                    f"Auto Rotation: {record.auto_rotation_deg:.1f}°",
                    f"Extra Rotation: {record.extra_rotation_deg}°",
                    f"Pain Score: {format_value(record.factor_values.get('Pain Score'))}",
                    f"Warnings: {'; '.join(record.warnings) if record.warnings else 'None'}",
                ]
            )
        )
        figure.tight_layout(pad=1.0)
        self.preview_canvas.draw_idle()

    def _run_statistics_async(self) -> None:
        if not self.results:
            return
        self._statistics_generation += 1
        generation = self._statistics_generation
        self.stats_status_var.set("Running MANOVA, PERMANOVA, and regression summaries...")
        self._set_status("Running exploratory shape statistics...")
        try:
            pc_count = int(self.stats_pc_count_var.get())
        except ValueError:
            pc_count = max(1, self.results.default_pc_count or 1)
        correction = self.stats_correction_var.get()

        def worker() -> None:
            try:
                rows = run_shape_statistics(self.results, pc_count, correction)
                self._enqueue_ui_task(lambda rows=rows: self._on_statistics_complete(generation, rows))
            except Exception as exc:
                self._enqueue_ui_task(lambda exc=exc: self._on_statistics_error(generation, exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_statistics_complete(self, generation: int, rows: List[Dict[str, Any]]) -> None:
        if generation != self._statistics_generation:
            return
        self.stats_rows = rows
        self.stats_tree.delete(*self.stats_tree.get_children())
        for index, row in enumerate(rows):
            self.stats_tree.insert(
                "",
                tk.END,
                iid=f"row-{index}",
                values=(
                    row["Factor"],
                    row["Factor Type"],
                    row["Test"],
                    row["Components"],
                    row["N"],
                    f"{row['Statistic']} = {format_value(row['Statistic Value'])}",
                    f"{row['Effect Size']} = {format_value(row['Effect Value'])}",
                    format_value(row["P Value"]),
                    format_value(row.get("Adjusted P")),
                ),
            )
        self.stats_status_var.set(
            f"{len(rows)} result rows computed with {self.stats_correction_var.get()} correction and {PERMANOVA_PERMUTATIONS} PERMANOVA permutations."
        )
        self._set_status("Exploratory statistics ready")

    def _on_statistics_error(self, generation: int, exc: Exception) -> None:
        if generation != self._statistics_generation:
            return
        self.stats_status_var.set("Statistics run failed.")
        self._set_status("Exploratory statistics failed")
        messagebox.showerror("Shape Statistics", str(exc))

    def _export_statistics_csv(self) -> None:
        if not self.stats_rows:
            messagebox.showinfo("Export CSV", "Run the statistics first.")
            return
        output_dir = os.path.join(self.project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        default_path = os.path.join(output_dir, "shape_analysis_statistics.csv")
        export_path = filedialog.asksaveasfilename(
            title="Export Shape Statistics CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(default_path),
            initialdir=os.path.dirname(default_path),
            filetypes=[("CSV Files", "*.csv")],
        )
        if not export_path:
            return
        with open(export_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.stats_rows)
        self.stats_status_var.set(f"Exported statistics to {export_path}")

    def _export_raw_shape_csv(self) -> None:
        if not self.raw_dataset or not self.raw_dataset.records:
            messagebox.showinfo("Export Raw CSV", "Load the shape-analysis cohort first.")
            return

        output_dir = os.path.join(self.project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        default_path = os.path.join(output_dir, "shape_analysis_raw_64pt.csv")
        export_path = filedialog.asksaveasfilename(
            title="Export Raw 64-Point Shape CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(default_path),
            initialdir=os.path.dirname(default_path),
            filetypes=[("CSV Files", "*.csv")],
        )
        if not export_path:
            return

        fieldnames = ["Patient ID", "Filename", "Laterality"]
        for index in range(RESAMPLE_POINTS):
            fieldnames.append(f"pt{index + 1:02d}_x")
            fieldnames.append(f"pt{index + 1:02d}_y")

        with open(export_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for raw_record in self.raw_dataset.records:
                resampled = resample_closed_contour(raw_record.raw_points_px, RESAMPLE_POINTS)
                flattened = resampled.reshape(-1)
                laterality = raw_record.factors.get("Original Laterality")
                if laterality not in {"Left", "Right"}:
                    laterality = ""

                filename = ""
                if raw_record.image_path:
                    filename = os.path.basename(raw_record.image_path)
                elif raw_record.label_path:
                    filename = os.path.splitext(os.path.basename(raw_record.label_path))[0] + ".png"

                row = {
                    "Patient ID": raw_record.empi or "",
                    "Filename": filename,
                    "Laterality": laterality,
                }
                for index, value in enumerate(flattened, start=1):
                    point_index = (index - 1) // 2 + 1
                    axis = "x" if index % 2 == 1 else "y"
                    row[f"pt{point_index:02d}_{axis}"] = f"{float(value):.6f}"
                writer.writerow(row)

        self._set_status(f"Exported raw 64-point CSV to {export_path}")
