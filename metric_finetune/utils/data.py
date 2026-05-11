from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class DepthSample:
    image_path: Path
    depth_path: Path
    mask_path: Path | None = None
    weight_path: Path | None = None


class MetricDepthDataset(Dataset):
    """Frame-level RGB/depth dataset for synthetic and registered bronchoscopy data."""

    def __init__(self, root: str | Path, image_size: tuple[int, int] = (224, 224)) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.samples = load_samples(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = load_rgb(sample.image_path, self.image_size)
        depth = load_array(sample.depth_path, self.image_size)
        mask = load_array(sample.mask_path, self.image_size) > 0.5 if sample.mask_path else depth > 0
        weight = load_array(sample.weight_path, self.image_size) if sample.weight_path else torch.ones_like(depth)
        return {
            "image": image,
            "depth": depth,
            "mask": mask,
            "weight": weight,
            "image_path": str(sample.image_path),
        }


def create_loader(
    root: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        MetricDepthDataset(root, image_size=image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_samples(root: Path) -> list[DepthSample]:
    manifest = root / "poses.csv"
    if manifest.exists():
        return load_manifest_samples(root, manifest)
    return load_directory_samples(root)


def load_manifest_samples(root: Path, manifest: Path) -> list[DepthSample]:
    samples: list[DepthSample] = []
    with manifest.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image_path = resolve(root, row["image_path"])
            depth_path = resolve(root, row["depth_path"])
            mask_path = optional_resolve(root, row.get("mask_path"))
            weight_path = optional_resolve(root, row.get("weight_path"))
            samples.append(DepthSample(image_path, depth_path, mask_path, weight_path))
    return samples


def load_directory_samples(root: Path) -> list[DepthSample]:
    image_dir = root / "images"
    depth_dir = root / "depths"
    mask_dir = root / "masks"
    weight_dir = root / "weights"
    samples: list[DepthSample] = []
    for image_path in sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))):
        depth_path = depth_dir / f"{image_path.stem}.npy"
        if not depth_path.exists():
            depth_path = depth_dir / f"{image_path.stem}.png"
        mask_path = first_existing(mask_dir, image_path.stem) if mask_dir.exists() else None
        weight_path = first_existing(weight_dir, image_path.stem) if weight_dir.exists() else None
        samples.append(DepthSample(image_path, depth_path, mask_path, weight_path))
    return samples


def load_rgb(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    if Image is None:
        raise ImportError("Pillow is required to load RGB images.")
    height, width = image_size
    image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def load_array(path: Path | None, image_size: tuple[int, int]) -> torch.Tensor:
    if path is None:
        raise ValueError("path is required")
    if path.suffix.lower() == ".npy":
        array = np.asarray(np.load(path), dtype=np.float32)
    else:
        if Image is None:
            raise ImportError("Pillow is required to load depth/mask images.")
        array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    tensor = torch.from_numpy(array).unsqueeze(0).to(dtype=torch.float32)
    if tuple(tensor.shape[-2:]) != tuple(image_size):
        tensor = F.interpolate(tensor.unsqueeze(0), size=image_size, mode="nearest").squeeze(0)
    return tensor.squeeze(0)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def optional_resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return resolve(root, value)


def first_existing(root: Path, stem: str) -> Path | None:
    for suffix in (".npy", ".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    return None

