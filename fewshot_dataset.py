import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF


IMAGE_DIR_CANDIDATES = ("images", "image", "imgs")
MASK_DIR_CANDIDATES = ("masks", "mask", "labels", "label", "gt")
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    mask_path: str
    name: str


def _resolve_split_dirs(root_dir: str, split: str) -> Tuple[str, str]:
    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_dir = None
    mask_dir = None
    for candidate in IMAGE_DIR_CANDIDATES:
        path = os.path.join(split_dir, candidate)
        if os.path.isdir(path):
            image_dir = path
            break
    for candidate in MASK_DIR_CANDIDATES:
        path = os.path.join(split_dir, candidate)
        if os.path.isdir(path):
            mask_dir = path
            break

    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not resolve image/mask directories under {split_dir}. "
            f"Looked for images in {IMAGE_DIR_CANDIDATES} and masks in {MASK_DIR_CANDIDATES}."
        )
    return image_dir, mask_dir


def _find_mask_path(mask_dir: str, image_name: str) -> str:
    base, ext = os.path.splitext(image_name)
    exact_path = os.path.join(mask_dir, image_name)
    if os.path.exists(exact_path):
        return exact_path

    for candidate_ext in (ext,) + VALID_EXTENSIONS:
        candidate = os.path.join(mask_dir, base + candidate_ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Mask file not found for {image_name} under {mask_dir}")


def _scan_split(root_dir: str, split: str) -> List[SampleRecord]:
    image_dir, mask_dir = _resolve_split_dirs(root_dir, split)
    image_names = sorted(
        name for name in os.listdir(image_dir) if name.lower().endswith(VALID_EXTENSIONS)
    )
    records: List[SampleRecord] = []
    for image_name in image_names:
        records.append(
            SampleRecord(
                image_path=os.path.join(image_dir, image_name),
                mask_path=_find_mask_path(mask_dir, image_name),
                name=image_name,
            )
        )
    if not records:
        raise RuntimeError(f"No images found in {image_dir}")
    return records


class FewShotSegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        shots: int = 1,
        image_size: int = 512,
        support_root: str | None = None,
        support_split: str | None = None,
        deterministic: bool = False,
        seed: int = 42,
        episodes_per_epoch: int = 0,
        keep_query_size: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.shots = shots
        self.image_size = image_size
        self.support_root = support_root or root_dir
        self.support_split = support_split or split
        self.deterministic = deterministic
        self.seed = seed
        self.episodes_per_epoch = episodes_per_epoch
        self.keep_query_size = keep_query_size

        self.query_records = _scan_split(root_dir, split)
        self.support_records = _scan_split(self.support_root, self.support_split)
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        if not self.deterministic and self.episodes_per_epoch > 0:
            return self.episodes_per_epoch
        return len(self.query_records)

    def _get_rng(self, idx: int) -> random.Random:
        return random.Random(self.seed + idx) if self.deterministic else random

    def _spawn_rng(self, parent_rng: random.Random, salt: int) -> random.Random:
        return random.Random(self.seed + salt) if self.deterministic else random.Random(parent_rng.random() + salt)

    def _sample_support_records(self, idx: int, query_record: SampleRecord) -> Sequence[SampleRecord]:
        rng = self._get_rng(idx)
        candidates = self.support_records
        if self.root_dir == self.support_root and self.split == self.support_split:
            candidates = [record for record in self.support_records if record.name != query_record.name]
            if not candidates:
                candidates = self.support_records

        if len(candidates) >= self.shots:
            return rng.sample(candidates, k=self.shots)
        return [candidates[rng.randrange(len(candidates))] for _ in range(self.shots)]

    def _pad_if_needed(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        width, height = image.size
        pad_width = max(0, self.image_size - width)
        pad_height = max(0, self.image_size - height)
        if pad_width == 0 and pad_height == 0:
            return image, mask

        padding = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        return TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=0)

    def _crop_pair(self, image: Image.Image, mask: Image.Image, rng: random.Random) -> Tuple[Image.Image, Image.Image]:
        image, mask = self._pad_if_needed(image, mask)
        width, height = image.size
        if self.deterministic:
            top = max(0, (height - self.image_size) // 2)
            left = max(0, (width - self.image_size) // 2)
        else:
            max_top = max(0, height - self.image_size)
            max_left = max(0, width - self.image_size)
            top = rng.randint(0, max_top) if max_top > 0 else 0
            left = rng.randint(0, max_left) if max_left > 0 else 0
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)
        return image, mask

    def _apply_train_augmentation(
        self, image: Image.Image, mask: Image.Image, rng: random.Random
    ) -> Tuple[Image.Image, Image.Image]:
        if self.deterministic:
            return image, mask
        if rng.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if rng.random() < 0.2:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def _load_pair(
        self,
        record: SampleRecord,
        rng: random.Random,
        preserve_size: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path).convert("L")
        original_size = image.size[::-1]

        if not preserve_size:
            image, mask = self._crop_pair(image, mask, rng)
        image, mask = self._apply_train_augmentation(image, mask, rng)
        image_tensor = self.to_tensor(image)
        mask_tensor = (self.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor, original_size

    def __getitem__(self, idx: int) -> dict:
        base_idx = idx % len(self.query_records)
        query_record = self.query_records[base_idx]
        episode_rng = self._get_rng(idx)
        query_rng = self._spawn_rng(episode_rng, base_idx * 2 + 1)
        query_image, query_mask, original_size = self._load_pair(
            query_record,
            query_rng,
            preserve_size=self.keep_query_size,
        )
        support_records = self._sample_support_records(idx, query_record)

        support_images = []
        support_masks = []
        for support_idx, support_record in enumerate(support_records):
            support_rng = self._spawn_rng(episode_rng, base_idx * 31 + support_idx + 17)
            support_image, support_mask, _ = self._load_pair(support_record, support_rng)
            support_images.append(support_image)
            support_masks.append(support_mask)

        return {
            "query_image": query_image,
            "query_mask": query_mask,
            "support_images": torch.stack(support_images, dim=0),
            "support_masks": torch.stack(support_masks, dim=0),
            "query_name": query_record.name,
            "original_size": torch.tensor(original_size, dtype=torch.long),
        }
