"""Small utilities shared by the dataset and training entry point."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


KNOWN_NORMALIZATION = {
    "NUAA-SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUDT-SIRST": {"mean": 107.80905151367188, "std": 33.02274703979492},
    "IRSTD-1K": {"mean": 87.4661865234375, "std": 39.71953201293945},
    "SIRST2": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "SIRST3": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUDT-SIRST-Sea": {"mean": 43.62403869628906, "std": 18.91838264465332},
    "IRDST-real": {"mean": 101.54053497314453, "std": 56.49856185913086},
}


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_crop(image, mask, patch_size, pos_prob=None):
    height, width = image.shape
    if min(height, width) < patch_size:
        pad_height = max(height, patch_size) - height
        pad_width = max(width, patch_size) - width
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant")
        mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")
        height, width = image.shape

    while True:
        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)
        image_patch = image[top:top + patch_size, left:left + patch_size]
        mask_patch = mask[top:top + patch_size, left:left + patch_size]
        if pos_prob is None or random.random() > pos_prob or mask_patch.sum() > 0:
            return image_patch, mask_patch


def Normalized(image, img_norm_cfg):
    std = max(float(img_norm_cfg["std"]), np.finfo(np.float32).eps)
    return (image - float(img_norm_cfg["mean"])) / std


def PadImg(image, times=32):
    height, width = image.shape
    pad_height = (-height) % times
    pad_width = (-width) % times
    return np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant")


def _find_image(image_dir: Path, image_id: str):
    for suffix in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = image_dir / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image found for '{image_id}' under {image_dir}")


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name in KNOWN_NORMALIZATION:
        return dict(KNOWN_NORMALIZATION[dataset_name])

    dataset_root = Path(dataset_dir) / dataset_name
    image_ids = []
    for split in ("trainval", "test"):
        index_path = dataset_root / "img_idx" / f"{split}.txt"
        if index_path.is_file():
            image_ids.extend(
                line.strip()
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if not image_ids:
        raise FileNotFoundError(f"No trainval/test index files found under {dataset_root / 'img_idx'}")

    means, stds = [], []
    for image_id in image_ids:
        image = np.asarray(Image.open(_find_image(dataset_root / "images", image_id)).convert("I"), dtype=np.float32)
        means.append(float(image.mean()))
        stds.append(float(image.std()))
    return {"mean": float(np.mean(means)), "std": float(np.mean(stds))}
