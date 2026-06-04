import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


CLASSIFIER_ARCH = "Torchvision Classifier"
DEFAULT_CLASSIFIER_BACKBONE = "efficientnet_v2_s"
DEFAULT_AUTO_POSITIVE_THRESHOLD = 0.95
DEFAULT_REVIEW_THRESHOLD = 0.75
CLASS_NAMES = ("non_fabella", "fabella")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


CLASSIFIER_BACKBONES = {
    "efficientnet_v2_s": {
        "label": "EfficientNet V2 S",
        "builder": models.efficientnet_v2_s,
        "weights_enum": models.EfficientNet_V2_S_Weights,
        "default_imgsz": 384,
    },
    "resnet50": {
        "label": "ResNet50",
        "builder": models.resnet50,
        "weights_enum": models.ResNet50_Weights,
        "default_imgsz": 224,
    },
    "resnet18": {
        "label": "ResNet18",
        "builder": models.resnet18,
        "weights_enum": models.ResNet18_Weights,
        "default_imgsz": 224,
    },
    "mobilenet_v3_small": {
        "label": "MobileNet V3 Small",
        "builder": models.mobilenet_v3_small,
        "weights_enum": models.MobileNet_V3_Small_Weights,
        "default_imgsz": 224,
    },
}


def get_backbone_info(backbone_key):
    if backbone_key not in CLASSIFIER_BACKBONES:
        raise ValueError(
            f"Unknown classifier backbone '{backbone_key}'. "
            f"Options: {list(CLASSIFIER_BACKBONES)}"
        )
    return CLASSIFIER_BACKBONES[backbone_key]


def default_imgsz_for_backbone(backbone_key):
    return get_backbone_info(backbone_key)["default_imgsz"]


def list_classifier_backbones():
    return list(CLASSIFIER_BACKBONES.keys())


def format_backbone_label(backbone_key):
    return get_backbone_info(backbone_key)["label"]


def load_png_as_pil(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    if img.dtype == np.uint16:
        img8 = (img / 256).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)

    img8 = cv2.normalize(img8, None, 0, 255, cv2.NORM_MINMAX)
    if len(img8.shape) == 2:
        img_rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def load_png_bgr_for_overlay(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img8 = (img / 256).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    img8 = cv2.normalize(img8, None, 0, 255, cv2.NORM_MINMAX)
    if len(img8.shape) == 2:
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    return img8


class ImagePathDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = load_png_as_pil(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, path


def gather_sorted_samples(pos_dir="data/sorted/pos", neg_dir="data/sorted/neg"):
    samples = []
    for folder, label in ((neg_dir, 0), (pos_dir, 1)):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".png"):
                samples.append((os.path.join(folder, name), label))
    return samples


def split_classifier_samples(samples, val_fraction=0.2, seed=42):
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    for label, label_samples in by_label.items():
        if not label_samples:
            raise ValueError(f"No samples found for class {label}.")
        rng.shuffle(label_samples)
        n_total = len(label_samples)
        n_val = max(1, int(round(n_total * val_fraction))) if n_total > 1 else 0
        n_val = min(n_val, n_total - 1) if n_total > 1 else 0
        val_samples.extend(label_samples[:n_val])
        train_samples.extend(label_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_classifier_transforms(imgsz, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_classifier_model(backbone_key, pretrained=True):
    info = get_backbone_info(backbone_key)
    weights = info["weights_enum"].DEFAULT if pretrained else None
    pretrained_loaded = bool(pretrained)
    warning = None
    try:
        model = info["builder"](weights=weights)
    except Exception as exc:
        model = info["builder"](weights=None)
        pretrained_loaded = False
        warning = (
            f"Could not load pretrained weights for {backbone_key}: {exc}. "
            "Continuing with randomly initialized weights."
        )

    if backbone_key.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)

    return model, pretrained_loaded, warning


def save_classifier_checkpoint(path, model, backbone_key, imgsz, extra=None):
    payload = {
        "arch": CLASSIFIER_ARCH,
        "backbone": backbone_key,
        "imgsz": imgsz,
        "class_names": CLASS_NAMES,
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_classifier_checkpoint(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    backbone_key = checkpoint.get("backbone", DEFAULT_CLASSIFIER_BACKBONE)
    imgsz = int(checkpoint.get("imgsz", default_imgsz_for_backbone(backbone_key)))
    model, _, _ = create_classifier_model(backbone_key, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint, device, build_classifier_transforms(imgsz, train=False)


@torch.no_grad()
def predict_fabella_probability(model, transform, image_path, device):
    img = load_png_as_pil(image_path)
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    return float(probs[1].item())
