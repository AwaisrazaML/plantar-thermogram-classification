"""Data discovery, subject-level splitting and transforms for plantar thermograms."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def discover_images(data_dir: str | Path) -> tuple[list[Path], list[int], list[str]]:
    """Find images in data_dir/<class_name>/*. Returns paths, labels, class names."""
    data_dir = Path(data_dir)
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise ValueError(f"Expected at least two class folders in {data_dir}, found {class_names}")

    paths, labels = [], []
    for label, name in enumerate(class_names):
        for path in sorted((data_dir / name).rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(label)
    if not paths:
        raise ValueError(f"No images found in {data_dir}")
    return paths, labels, class_names


def subject_ids(
    paths: list[Path], labels: list[int], pattern: str | None, data_dir: str | Path | None = None
) -> list[str]:
    """Subject ID per image, so one person's images never cross splits.

    Priority: regex on the file name, then the per-subject sub-folder
    (data/<class>/<subject>/image.png), otherwise each image on its own.
    """
    regex = re.compile(pattern) if pattern else None
    ids = []
    for path, label in zip(paths, labels):
        subject = path.stem
        if regex:
            match = regex.search(path.stem)
            subject = match.group(1) if match else path.stem
        elif data_dir is not None:
            relative = path.relative_to(data_dir).parts  # (class, [subject,] file)
            if len(relative) >= 3:
                subject = relative[1]
            else:
                subject = str(path)
        else:
            subject = str(path)
        # prefix with label so IDs are unique across classes
        ids.append(f"{label}:{subject}")
    return ids


def split_by_subject(
    paths: list[Path],
    labels: list[int],
    subjects: list[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[list[Path], list[int]]]:
    """Stratified split at subject level so one person's images stay in one split."""
    subject_to_label: dict[str, int] = {}
    for subject, label in zip(subjects, labels):
        subject_to_label.setdefault(subject, label)
    unique = np.array(list(subject_to_label))
    unique_labels = np.array([subject_to_label[s] for s in unique])

    train_val, test = train_test_split(
        unique, test_size=test_size, stratify=unique_labels, random_state=seed
    )
    train_val_labels = np.array([subject_to_label[s] for s in train_val])
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=seed,
    )

    groups = {"train": set(train), "val": set(val), "test": set(test)}
    splits: dict[str, tuple[list[Path], list[int]]] = {k: ([], []) for k in groups}
    for path, label, subject in zip(paths, labels, subjects):
        for name, members in groups.items():
            if subject in members:
                splits[name][0].append(path)
                splits[name][1].append(label)
    return splits


def build_transforms(image_size: int = 224) -> dict[str, transforms.Compose]:
    """Augmentation for training, plain resize for evaluation."""
    train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluate = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return {"train": train, "eval": evaluate}


class ThermogramDataset(Dataset):
    """Loads thermograms as 3-channel RGB so ImageNet backbones can be reused."""

    def __init__(self, paths: list[Path], labels: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
