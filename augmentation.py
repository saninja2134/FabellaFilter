# Augmentation engine using OpenCV and NumPy. No albumentations required.
import cv2
import numpy as np
import random
import math
import os
import copy


# ─────────────────────────────────────────────────────────────────
# UTILITY: Label handling
# ─────────────────────────────────────────────────────────────────

def read_labels(label_path):
    # Returns list of (class_id, [x1,y1, x2,y2, ... xn,yn]) tuples.
    labels = []
    if not label_path or not os.path.exists(label_path):
        return labels
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    cid = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    labels.append((cid, coords))
    except Exception:
        pass
    return labels


def write_labels(label_path, labels):
    # Writes label list to file.
    with open(label_path, 'w') as f:
        for cid, coords in labels:
            coords_clamped = [max(0.0, min(1.0, c)) for c in coords]
            f.write(f"{cid} {' '.join(f'{c:.6f}' for c in coords_clamped)}\n")


def pair_points(coords):
    # Converts flat coord list to list of (x, y) tuples.
    return [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]


def flat_points(pts):
    # Converts (x, y) tuples back to flat list.
    return [v for pt in pts for v in pt]


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────

def _resize_stretch(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_padded(img, w, h, pad_val=0):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if len(img.shape) == 3:
        canvas = np.full((h, w, img.shape[2]), pad_val, dtype=img.dtype)
    else:
        canvas = np.full((h, w), pad_val, dtype=img.dtype)
    y_off = (h - nh) // 2
    x_off = (w - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas


def _resize_crop(img, w, h):
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y_off = (nh - h) // 2
    x_off = (nw - w) // 2
    return resized[y_off:y_off+h, x_off:x_off+w]


def apply_preprocessing(img, config):
    # Apply all preprocessing steps to an image.
    pre = config.get('preprocessing', {})

    # Resize
    resize = pre.get('resize', {})
    w, h = resize.get('width', 1024), resize.get('height', 1024)
    mode = resize.get('mode', 'Stretch')
    if mode == 'Stretch':
        img = _resize_stretch(img, w, h)
    elif mode == 'Fit (Black Pad)':
        img = _resize_padded(img, w, h, pad_val=0)
    elif mode == 'Fit (White Pad)':
        img = _resize_padded(img, w, h, pad_val=255)
    elif mode == 'Crop':
        img = _resize_crop(img, w, h)

    # Auto-Contrast (CLAHE)
    if pre.get('auto_contrast', False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img.shape) == 2:
            img = clahe.apply(img if img.dtype == np.uint8 else (img // 256).astype(np.uint8))
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Grayscale
    if pre.get('grayscale', False):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


# ─────────────────────────────────────────────────────────────────
# AUGMENTATION FUNCTIONS (image + label transforms)
# ─────────────────────────────────────────────────────────────────

def _transform_labels_geometric(labels, transform_fn, filter_oob=True):
    # Apply a point transformation function to all label coordinates.
    new_labels = []
    for cid, coords in labels:
        pts = pair_points(coords)
        new_pts = [transform_fn(x, y) for x, y in pts]
        if filter_oob:
            if any(0 <= x <= 1 and 0 <= y <= 1 for x, y in new_pts):
                new_labels.append((cid, flat_points(new_pts)))
        else:
            new_labels.append((cid, flat_points(new_pts)))
    return new_labels


def aug_flip(img, labels, horizontal=True, vertical=False):
    if horizontal:
        img = cv2.flip(img, 1)
        labels = _transform_labels_geometric(labels, lambda x, y: (1.0 - x, y), filter_oob=False)
    if vertical:
        img = cv2.flip(img, 0)
        labels = _transform_labels_geometric(labels, lambda x, y: (x, 1.0 - y), filter_oob=False)
    return img, labels


def aug_rotate90(img, labels, clockwise=False, counter_clockwise=False, upside_down=False):
    choices = []
    if clockwise:       choices.append(cv2.ROTATE_90_CLOCKWISE)
    if counter_clockwise: choices.append(cv2.ROTATE_90_COUNTERCLOCKWISE)
    if upside_down:     choices.append(cv2.ROTATE_180)
    if not choices:     return img, labels

    code = random.choice(choices)
    img = cv2.rotate(img, code)

    # Normalized point transformation for each rotation
    if code == cv2.ROTATE_90_CLOCKWISE:
        fn = lambda x, y: (1.0 - y, x)
    elif code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        fn = lambda x, y: (y, 1.0 - x)
    else:  # 180
        fn = lambda x, y: (1.0 - x, 1.0 - y)

    labels = _transform_labels_geometric(labels, fn, filter_oob=False)
    return img, labels


def aug_rotation(img, labels, angle_range=(-45, 45)):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    # Compute bounding box of rotated image
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2

    img = cv2.warpAffine(img, M, (nw, nh),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    img = cv2.resize(img, (w, h))

    # Transform labels
    rad = math.radians(-angle)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    scale_x, scale_y = w / nw, h / nh

    def rotate_pt(x, y):
        px = x * nw - nw / 2
        py = y * nh - nh / 2
        rx = px * cos_r - py * sin_r + nw / 2
        ry = px * sin_r + py * cos_r + nh / 2
        return rx * scale_x / w, ry * scale_y / h

    labels = _transform_labels_geometric(labels, lambda x, y: rotate_pt(x, y))
    return img, labels


def aug_crop(img, labels, zoom_range=(0, 50)):
    h, w = img.shape[:2]
    zoom_pct = random.uniform(zoom_range[0], zoom_range[1]) / 100.0
    margin = zoom_pct / 2.0

    x1n = random.uniform(0, margin)
    y1n = random.uniform(0, margin)
    x2n = random.uniform(1.0 - margin, 1.0)
    y2n = random.uniform(1.0 - margin, 1.0)

    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
    if x2 <= x1 or y2 <= y1:
        return img, labels

    crop_w = x2 - x1
    crop_h = y2 - y1
    img = img[y1:y2, x1:x2]
    img = cv2.resize(img, (w, h))

    def crop_pt(x, y):
        nx = (x - x1n) / (x2n - x1n)
        ny = (y - y1n) / (y2n - y1n)
        return nx, ny

    labels = _transform_labels_geometric(labels, lambda x, y: crop_pt(x, y))
    return img, labels


def aug_shear(img, labels, horizontal=15.0, vertical=15.0):
    h, w = img.shape[:2]
    sh_h = math.tan(math.radians(horizontal * random.uniform(-1, 1)))
    sh_v = math.tan(math.radians(vertical * random.uniform(-1, 1)))

    M = np.float32([[1, sh_h, -sh_h * h / 2],
                    [sh_v, 1, -sh_v * w / 2]])
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    def shear_pt(x, y):
        px = x * w
        py = y * h
        nx = (px + sh_h * py - sh_h * h / 2) / w
        ny = (sh_v * px + py - sh_v * w / 2) / h
        return nx, ny

    labels = _transform_labels_geometric(labels, lambda x, y: shear_pt(x, y))
    return img, labels


def _ensure_bgr_uint8(img):
    # Ensure image is BGR uint8 for color transforms.
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def aug_brightness(img, labels, brightness_range=(-25, 25)):
    img = _ensure_bgr_uint8(img)
    factor = random.uniform(brightness_range[0], brightness_range[1]) / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + factor), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, labels


def aug_exposure(img, labels, exposure_range=(-15, 15)):
    img = _ensure_bgr_uint8(img)
    factor = random.uniform(exposure_range[0], exposure_range[1]) / 100.0
    img = np.clip(img.astype(np.float32) * (2.0 ** factor), 0, 255).astype(np.uint8)
    return img, labels


def aug_saturation(img, labels, saturation_range=(-25, 25)):
    img = _ensure_bgr_uint8(img)
    factor = random.uniform(saturation_range[0], saturation_range[1]) / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + factor), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, labels


def aug_hue(img, labels, hue_range=(-15, 15)):
    img = _ensure_bgr_uint8(img)
    shift = random.uniform(hue_range[0], hue_range[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(shift / 2)) % 180  # OpenCV H range 0-179
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, labels


def aug_blur(img, labels, max_px=3):
    ksize = int(random.uniform(1, max_px))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img, labels


def aug_noise(img, labels, max_pct=5.0):
    img8 = _ensure_bgr_uint8(img)
    noise_amount = random.uniform(0, max_pct) / 100.0
    n_pixels = int(img8.shape[0] * img8.shape[1] * noise_amount)
    ys = np.random.randint(0, img8.shape[0], n_pixels)
    xs = np.random.randint(0, img8.shape[1], n_pixels)
    img8[ys, xs] = np.random.randint(0, 256, (n_pixels, img8.shape[2]) if len(img8.shape) == 3 else (n_pixels,), dtype=np.uint8)
    return img8, labels


def aug_mosaic(img, labels, all_samples):
    # Combine 4 images into a 2x2 mosaic. all_samples: list of (img, labels) pairs.
    if len(all_samples) < 3:
        return img, labels

    h, w = img.shape[:2]
    half_h, half_w = h // 2, w // 2

    # Pick 3 random additional samples
    chosen = random.sample(all_samples, min(3, len(all_samples)))
    tiles = [(img, labels)] + chosen

    mosaic_img = np.zeros((h, w) + ((img.shape[2],) if len(img.shape) == 3 else ()), dtype=img.dtype)
    mosaic_labels = []

    positions = [(0, 0), (0, half_w), (half_h, 0), (half_h, half_w)]
    scale_pairs = [(0.0, 0.0, 0.5, 0.5),
                   (0.5, 0.0, 1.0, 0.5),
                   (0.0, 0.5, 0.5, 1.0),
                   (0.5, 0.5, 1.0, 1.0)]

    for i, ((t_img, t_labels), (ys, xs, ye, xe)) in enumerate(zip(tiles, scale_pairs)):
        row, col = positions[i]
        tile = cv2.resize(t_img, (half_w, half_h))
        mosaic_img[row:row+half_h, col:col+half_w] = tile

        # Re-normalize labels into mosaic quadrant
        for cid, coords in t_labels:
            pts = pair_points(coords)
            new_pts = [(xs + (xe - xs) * x, ys + (ye - ys) * y) for x, y in pts]
            mosaic_labels.append((cid, flat_points(new_pts)))

    return mosaic_img, mosaic_labels


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def augment_sample(img, labels, config, all_mosaic_samples=None):
    # Apply all enabled augmentations based on config.
    augs = config.get('augmentations', {})

    # Flip
    f = augs.get('flip', {})
    if f.get('enabled'):
        img, labels = aug_flip(img, labels,
                                horizontal=f.get('horizontal', True),
                                vertical=f.get('vertical', False))

    # 90° Rotate
    r90 = augs.get('rotate90', {})
    if r90.get('enabled'):
        img, labels = aug_rotate90(img, labels,
                                   clockwise=r90.get('clockwise', True),
                                   counter_clockwise=r90.get('counter_clockwise', True),
                                   upside_down=r90.get('upside_down', True))

    # Arbitrary rotation
    rot = augs.get('rotation', {})
    if rot.get('enabled'):
        img, labels = aug_rotation(img, labels, angle_range=rot.get('range', [-45, 45]))

    # Crop
    crop = augs.get('crop', {})
    if crop.get('enabled'):
        img, labels = aug_crop(img, labels, zoom_range=crop.get('range', [0, 50]))

    # Shear
    shear = augs.get('shear', {})
    if shear.get('enabled'):
        img, labels = aug_shear(img, labels,
                                horizontal=shear.get('horizontal', 15),
                                vertical=shear.get('vertical', 15))

    # Brightness
    bri = augs.get('brightness', {})
    if bri.get('enabled'):
        img, labels = aug_brightness(img, labels, brightness_range=bri.get('range', [-25, 25]))

    # Exposure
    exp = augs.get('exposure', {})
    if exp.get('enabled'):
        img, labels = aug_exposure(img, labels, exposure_range=exp.get('range', [-15, 15]))

    # Saturation
    sat = augs.get('saturation', {})
    if sat.get('enabled'):
        img, labels = aug_saturation(img, labels, saturation_range=sat.get('range', [-25, 25]))

    # Hue
    hue = augs.get('hue', {})
    if hue.get('enabled'):
        img, labels = aug_hue(img, labels, hue_range=hue.get('range', [-15, 15]))

    # Blur
    blur = augs.get('blur', {})
    if blur.get('enabled'):
        img, labels = aug_blur(img, labels, max_px=blur.get('max_px', 3))

    # Noise
    noise = augs.get('noise', {})
    if noise.get('enabled'):
        img, labels = aug_noise(img, labels, max_pct=noise.get('max_pct', 5))

    # Mosaic
    mosaic = augs.get('mosaic', {})
    if mosaic.get('enabled') and all_mosaic_samples:
        img, labels = aug_mosaic(img, labels, all_mosaic_samples)

    return img, labels
