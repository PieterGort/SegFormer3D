import json
import os
import random
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class NnUNetSegDataset(Dataset):
    """
    Generic nnU-Net style dataset loader for 3D segmentation tasks.

    Expected layout:
      root_dir/
        dataset.json
        imagesTr/<case>_0000.nii.gz
        labelsTr/<case>.nii.gz
    """

    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        transform: Optional[Any] = None,
        fold_id: Optional[int] = None,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        image_dir: str = "imagesTr",
        label_dir: str = "labelsTr",
        dataset_json: str = "dataset.json",
        clip_percentiles: Optional[Sequence[float]] = (0.5, 99.5),
    ) -> None:
        super().__init__()
        del fold_id  # Reserved for future k-fold support.

        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.image_dir = os.path.join(root_dir, image_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.clip_percentiles = self._normalize_percentiles(clip_percentiles)

        dataset_json_fp = os.path.join(root_dir, dataset_json)
        self.dataset_meta = self._load_dataset_meta(dataset_json_fp)
        self.file_ending = self.dataset_meta.get("file_ending", ".nii.gz")
        self.expected_num_channels = len(self.dataset_meta.get("channel_names", {})) or None

        label_paths = sorted(glob(os.path.join(self.label_dir, f"*{self.file_ending}")))
        if not label_paths:
            raise FileNotFoundError(
                f"No label files ending with '{self.file_ending}' were found in {self.label_dir}"
            )

        case_ids = [self._strip_suffix(os.path.basename(path), self.file_ending) for path in label_paths]
        split_case_ids = self._split_case_ids(case_ids, val_fraction=val_fraction, split_seed=split_seed)
        self.case_ids = split_case_ids["train"] if is_train else split_case_ids["val"]

        if not self.case_ids:
            split_name = "train" if is_train else "validation"
            raise ValueError(f"No cases assigned to the {split_name} split for {root_dir}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id = self.case_ids[idx]
        image = self._load_case_image(case_id)
        label = self._load_label(case_id)

        data = {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).long().unsqueeze(0),
        }

        if self.transform:
            data = self.transform(data)

        if isinstance(data, list):
            for sample in data:
                sample["image"] = sample["image"].float()
                sample["label"] = sample["label"].long()
            return data

        data["image"] = data["image"].float()
        data["label"] = data["label"].long()
        return data

    def _load_case_image(self, case_id: str) -> np.ndarray:
        image_paths = sorted(glob(os.path.join(self.image_dir, f"{case_id}_*{self.file_ending}")))
        if not image_paths:
            raise FileNotFoundError(f"No image channels found for case '{case_id}' in {self.image_dir}")

        if self.expected_num_channels is not None and len(image_paths) != self.expected_num_channels:
            raise ValueError(
                f"Expected {self.expected_num_channels} channels for case '{case_id}', "
                f"found {len(image_paths)}"
            )

        channels = [self._load_nifti(image_path) for image_path in image_paths]
        image = np.stack(channels, axis=0)
        return self._normalize_image(image)

    def _load_label(self, case_id: str) -> np.ndarray:
        label_path = os.path.join(self.label_dir, f"{case_id}{self.file_ending}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found for case '{case_id}': {label_path}")

        label = self._load_nifti(label_path)
        return np.rint(label).astype(np.int64)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        normalized_channels: List[np.ndarray] = []
        for channel in image:
            channel = channel.astype(np.float32, copy=False)

            if self.clip_percentiles is not None:
                lower, upper = np.percentile(channel, self.clip_percentiles)
                channel = np.clip(channel, lower, upper)

            mean = float(channel.mean())
            std = float(channel.std())
            if std < 1e-6:
                std = 1.0

            normalized_channels.append((channel - mean) / std)

        return np.stack(normalized_channels, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _load_dataset_meta(dataset_json_fp: str) -> Dict[str, Any]:
        if not os.path.exists(dataset_json_fp):
            return {}

        with open(dataset_json_fp, "r", encoding="utf-8") as infile:
            return json.load(infile)

    @staticmethod
    def _load_nifti(path: str) -> np.ndarray:
        return np.asarray(nib.load(path).get_fdata(dtype=np.float32), dtype=np.float32)

    @staticmethod
    def _normalize_percentiles(
        clip_percentiles: Optional[Sequence[float]],
    ) -> Optional[Tuple[float, float]]:
        if clip_percentiles in (None, "None"):
            return None

        if len(clip_percentiles) != 2:
            raise ValueError(
                "clip_percentiles must be a two-item sequence like [0.5, 99.5] or null"
            )

        lower, upper = float(clip_percentiles[0]), float(clip_percentiles[1])
        if lower >= upper:
            raise ValueError("clip_percentiles lower bound must be smaller than upper bound")
        return lower, upper

    @staticmethod
    def _split_case_ids(
        case_ids: Sequence[str],
        val_fraction: float,
        split_seed: int,
    ) -> Dict[str, List[str]]:
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in the range [0.0, 1.0)")

        shuffled_case_ids = list(case_ids)
        random.Random(split_seed).shuffle(shuffled_case_ids)

        if len(shuffled_case_ids) == 1 or val_fraction == 0.0:
            return {"train": shuffled_case_ids, "val": shuffled_case_ids}

        num_val = max(1, int(round(len(shuffled_case_ids) * val_fraction)))
        num_val = min(num_val, len(shuffled_case_ids) - 1)

        return {
            "train": shuffled_case_ids[num_val:],
            "val": shuffled_case_ids[:num_val],
        }

    @staticmethod
    def _strip_suffix(filename: str, suffix: str) -> str:
        if not filename.endswith(suffix):
            raise ValueError(f"Expected filename '{filename}' to end with '{suffix}'")
        return filename[: -len(suffix)]
