"""Dataset loaders for infrared small-target segmentation datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import Normalized, PadImg, get_img_norm_cfg, random_crop


IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def _read_index(dataset_root: Path, split: str):
    index_path = dataset_root / "img_idx" / f"{split}.txt"
    if not index_path.is_file():
        raise FileNotFoundError(f"Dataset split file not found: {index_path}")
    return [line.strip() for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_file(directory: Path, image_id: str) -> Path:
    candidate = directory / image_id
    if candidate.is_file():
        return candidate
    for suffix in IMAGE_EXTENSIONS:
        candidate = directory / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for '{image_id}' under {directory}")


def _load_pair(dataset_root: Path, image_id: str):
    image_path = _find_file(dataset_root / "images", image_id)
    mask_path = _find_file(dataset_root / "masks", image_id)
    image = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    return image, mask


class TrainSetLoader(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_name,
        patch_size,
        img_norm_cfg=None,
        split="trainval",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_dir) / dataset_name
        self.patch_size = int(patch_size)
        self.train_list = _read_index(self.dataset_dir, split)
        self.img_norm_cfg = img_norm_cfg or get_img_norm_cfg(dataset_name, dataset_dir)

    def __getitem__(self, index):
        image, mask = _load_pair(self.dataset_dir, self.train_list[index])
        image = Normalized(image, self.img_norm_cfg)
        image, mask = random_crop(image, mask, self.patch_size, pos_prob=0.5)

        if np.random.random() < 0.5:
            image, mask = image[::-1, :], mask[::-1, :]
        if np.random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if np.random.random() < 0.5:
            image, mask = image.T, mask.T

        image = torch.from_numpy(np.ascontiguousarray(image[None, ...])).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()
        return image, mask

    def __len__(self):
        return len(self.train_list)


class TestSetLoader(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_name,
        split="test",
        img_norm_cfg=None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_dir) / dataset_name
        self.test_list = _read_index(self.dataset_dir, split)
        self.img_norm_cfg = img_norm_cfg or get_img_norm_cfg(dataset_name, dataset_dir)

    def __getitem__(self, index):
        image_id = self.test_list[index]
        image, mask = _load_pair(self.dataset_dir, image_id)
        image = Normalized(image, self.img_norm_cfg)
        height, width = image.shape
        image, mask = PadImg(image), PadImg(mask)

        image = torch.from_numpy(np.ascontiguousarray(image[None, ...])).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()
        return image, mask, (height, width), image_id

    def __len__(self):
        return len(self.test_list)
