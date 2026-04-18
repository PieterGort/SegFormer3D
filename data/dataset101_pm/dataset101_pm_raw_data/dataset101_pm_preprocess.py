import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import nibabel
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import EnsureType, Orientation, Spacing
from tqdm import tqdm


# Preprocessing format version.  Stored as meta.json inside the save_dir so the
# launcher can detect stale output (e.g. the legacy one-hot+MinMax layout) and
# re-run preprocessing automatically.
PREPROCESS_VERSION = 2


class Dataset101PMPreprocess:
    """Preprocess Dataset101_PM (nnU-Net style CT) into fast-loading .pt tensors.

    Output layout (v2):
      save_dir/
        <case>/<case>_modalities.pt   # float32 (C, W, H, D) image tensor
        <case>/<case>_label.pt        # uint8  (1, W, H, D) integer label map
        meta.json                     # preprocessing metadata + version marker

    Key differences vs v1 (which produced the 21916679 / 21929592 runs):
      * Labels are saved as an integer label map, NOT one-hot.  This lets
        downstream augmentations use RandCropByPosNegLabeld (foreground-biased
        sampling) and side-steps the v1 bug where center_crop padding left
        regions with all-zero one-hot channels.
      * Image normalization: HU-clip to [-175, 250] (safety, data is already
        clipped upstream) + per-case z-score.  v1 used MinMaxScaler over the
        full volume, which is brittle on CT when extremes sneak through.
    """

    def __init__(
        self,
        dataset_root: str,
        save_dir: str,
        image_dir: str = "imagesTr",
        label_dir: str = "labelsTr",
        dataset_json: str = "dataset.json",
        target_shape: Tuple[int, int, int] = (192, 192, 256),
        target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        hu_clip: Optional[Tuple[float, float]] = (-175.0, 250.0),
    ) -> None:
        self.dataset_root = os.path.abspath(dataset_root)
        self.image_dir = os.path.join(self.dataset_root, image_dir)
        self.label_dir = os.path.join(self.dataset_root, label_dir)
        self.dataset_meta = self._load_dataset_meta(os.path.join(self.dataset_root, dataset_json))
        self.file_ending = self.dataset_meta.get("file_ending", ".nii.gz")
        self.channel_codes = self._get_channel_codes(self.dataset_meta)
        self.label_values = self._get_label_values(self.dataset_meta)
        self.num_classes = len(self.label_values)
        self.target_shape = tuple(int(dim) for dim in target_shape)
        self.target_spacing = tuple(float(s) for s in target_spacing)
        self.hu_clip = None if hu_clip is None else (float(hu_clip[0]), float(hu_clip[1]))
        self.save_dir = os.path.abspath(save_dir)

        assert os.path.exists(self.image_dir), f"Image directory not found: {self.image_dir}"
        assert os.path.exists(self.label_dir), f"Label directory not found: {self.label_dir}"

        self.case_names = sorted(
            self.remove_case_name_artifact(case_name)
            for case_name in next(os.walk(self.label_dir), (None, None, []))[2]
            if case_name.endswith(self.file_ending)
        )

    def __len__(self) -> int:
        return len(self.case_names)

    @staticmethod
    def _load_dataset_meta(dataset_json_fp: str) -> Dict:
        if not os.path.exists(dataset_json_fp):
            return {}

        with open(dataset_json_fp, "r", encoding="utf-8") as infile:
            return json.load(infile)

    @staticmethod
    def _get_channel_codes(dataset_meta: Dict) -> List[str]:
        channel_names = dataset_meta.get("channel_names", {})
        if channel_names:
            return [f"{int(channel_code):04d}" for channel_code in sorted(channel_names, key=int)]
        return ["0000"]

    @staticmethod
    def _get_label_values(dataset_meta: Dict) -> List[int]:
        labels = dataset_meta.get("labels", {})
        if labels:
            return sorted(int(label_value) for label_value in labels.values())
        return [0, 1]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """HU-clip (safety) + per-case z-score normalization.

        The nnU-Net CT recipe clips to a dataset-specific HU window and then
        applies z-score with per-case mean/std.  Dataset101_PM is already
        clipped upstream to [-175, 250] HU, but we re-enforce the clip here to
        be robust to accidental outliers.
        """
        x = x.astype(np.float32, copy=False)
        if self.hu_clip is not None:
            x = np.clip(x, self.hu_clip[0], self.hu_clip[1])
        mean = float(x.mean())
        std = float(x.std())
        if std < 1e-6:
            std = 1.0
        return ((x - mean) / std).astype(np.float32, copy=False)

    @staticmethod
    def orient(x: MetaTensor) -> MetaTensor:
        return Orientation(axcodes="RAS")(x)

    def resample(self, x: MetaTensor, mode: str = "bilinear") -> MetaTensor:
        return Spacing(pixdim=self.target_spacing, mode=mode)(x)

    @staticmethod
    def detach_meta(x: MetaTensor) -> np.ndarray:
        return EnsureType(data_type="numpy", track_meta=False)(x)

    @staticmethod
    def remove_case_name_artifact(case_name: str) -> str:
        if case_name.endswith(".nii.gz"):
            return case_name[: -len(".nii.gz")]
        return case_name.rsplit(".", 1)[0]

    def get_modality_fp(self, case_name: str, folder: str, channel_code: str = None) -> str:
        if channel_code is None:
            file_name = f"{case_name}{self.file_ending}"
        else:
            file_name = f"{case_name}_{channel_code}{self.file_ending}"
        return os.path.join(self.dataset_root, folder, file_name)

    @staticmethod
    def load_nifti(fp: str) -> Tuple[np.ndarray, np.ndarray]:
        nifti_data = nibabel.load(fp)
        nifti_scan = nifti_data.get_fdata()
        affine = nifti_data.affine
        return nifti_scan, affine

    @staticmethod
    def _crop_to_foreground(
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        foreground_mask = image[0] > image[0].min()
        if not np.any(foreground_mask):
            return image, label

        coords = np.where(foreground_mask)
        spatial_slices = tuple(
            slice(int(coord.min()), int(coord.max()) + 1) for coord in coords
        )
        return (
            image[(slice(None),) + spatial_slices],
            label[(slice(None),) + spatial_slices],
        )

    def _center_crop_or_pad(self, array: np.ndarray) -> np.ndarray:
        output = np.zeros((array.shape[0],) + self.target_shape, dtype=array.dtype)

        src_slices = []
        dst_slices = []
        for axis, target_dim in enumerate(self.target_shape, start=1):
            current_dim = array.shape[axis]
            if current_dim >= target_dim:
                start = (current_dim - target_dim) // 2
                src_slices.append(slice(start, start + target_dim))
                dst_slices.append(slice(0, target_dim))
            else:
                pad_before = (target_dim - current_dim) // 2
                src_slices.append(slice(0, current_dim))
                dst_slices.append(slice(pad_before, pad_before + current_dim))

        output[(slice(None),) + tuple(dst_slices)] = array[(slice(None),) + tuple(src_slices)]
        return output

    def preprocess_modality(self, data_fp: str) -> np.ndarray:
        data, affine = self.load_nifti(data_fp)
        data = self.normalize(x=data)
        data = data[np.newaxis, ...]
        data = MetaTensor(x=data, affine=affine)
        data = self.orient(data)
        data = self.resample(data, mode="bilinear")
        return self.detach_meta(data)

    def preprocess_label(self, data_fp: str) -> np.ndarray:
        """Return the resampled label as an integer class-id map.

        Shape: (1, D, H, W), dtype uint8.  We intentionally do NOT one-hot
        encode here — the padding step in __getitem__ needs to distinguish
        background (class 0) from "no class assigned", and downstream
        augmentations (RandCropByPosNegLabeld) expect a label map anyway.
        """
        data, affine = self.load_nifti(data_fp)
        data = np.rint(data).astype(np.float32, copy=False)
        data = data[np.newaxis, ...]  # (1, D, H, W)
        data = MetaTensor(x=data, affine=affine)
        data = self.orient(data)
        data = self.resample(data, mode="nearest")
        data = self.detach_meta(data)          # (1, D, H, W) numpy float32
        return np.rint(data).astype(np.uint8, copy=False)

    def __getitem__(self, idx: int):
        case_name = self.case_names[idx]

        modalities = []
        for channel_code in self.channel_codes:
            modality_fp = self.get_modality_fp(case_name, "imagesTr", channel_code)
            modality = self.preprocess_modality(modality_fp)
            modalities.append(modality)

        label_fp = self.get_modality_fp(case_name, "labelsTr", None)
        label = self.preprocess_label(label_fp)

        modalities = np.concatenate(modalities, axis=0, dtype=np.float32)
        modalities, label = self._crop_to_foreground(modalities, label)
        modalities = self._center_crop_or_pad(modalities)
        # Label is padded with zeros which, in label-map form, correctly marks
        # those voxels as background (class 0).
        label = self._center_crop_or_pad(label)

        # swapaxes(1, 3): (C, D, H, W) → (C, W, H, D) — transverse plane, matching BraTS convention.
        modalities = modalities.swapaxes(1, 3)
        label = label.swapaxes(1, 3)
        return modalities.astype(np.float32, copy=False), label.astype(np.uint8, copy=False), case_name

    def __call__(self) -> None:
        num_workers = self._resolve_num_workers()
        print("started preprocessing Dataset101_PM (v%d)..." % PREPROCESS_VERSION)
        print(f"using {num_workers} worker processes")
        print(f"target spacing: {self.target_spacing} mm  |  target shape: {self.target_shape}")
        print(f"hu clip: {self.hu_clip}  |  label format: integer label map")
        os.makedirs(self.save_dir, exist_ok=True)
        with Pool(processes=num_workers) as multi_p:
            for _ in tqdm(
                multi_p.imap_unordered(self.process, range(self.__len__())),
                total=self.__len__(),
                desc="preprocess",
            ):
                pass
        self._write_meta_json()
        print("finished preprocessing Dataset101_PM...")

    def _write_meta_json(self) -> None:
        meta = {
            "preprocess_version": PREPROCESS_VERSION,
            "label_format": "label_map",
            "num_classes": self.num_classes,
            "label_values": list(self.label_values),
            "target_spacing": list(self.target_spacing),
            "target_shape": list(self.target_shape),
            "hu_clip": None if self.hu_clip is None else list(self.hu_clip),
            "normalization": "hu_clip + per-case z-score",
        }
        with open(os.path.join(self.save_dir, "meta.json"), "w", encoding="utf-8") as outfile:
            json.dump(meta, outfile, indent=2)

    def process(self, idx: int) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        modalities, label, case_name = self.__getitem__(idx)
        data_save_path = os.path.join(self.save_dir, case_name)
        os.makedirs(data_save_path, exist_ok=True)
        torch.save(torch.from_numpy(modalities), os.path.join(data_save_path, f"{case_name}_modalities.pt"))
        torch.save(torch.from_numpy(label), os.path.join(data_save_path, f"{case_name}_label.pt"))
        return case_name

    @staticmethod
    def _resolve_num_workers() -> int:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                return max(1, min(int(slurm_cpus), 8))
            except ValueError:
                pass
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, 8))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Dataset101_PM into SegFormer3D tensors.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM",
        help="Path to the nnU-Net style Dataset101_PM root.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed_2mm",
        help="Directory where preprocessed case folders will be saved.",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(192, 192, 256),
        metavar=("D", "H", "W"),
        help="Final spatial shape saved for each case (before axis swap).",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 2.0),
        metavar=("X", "Y", "Z"),
        help="Isotropic voxel spacing in mm to resample to before cropping.",
    )
    parser.add_argument(
        "--hu-clip",
        type=float,
        nargs=2,
        default=(-175.0, 250.0),
        metavar=("LOWER", "UPPER"),
        help="HU clipping range applied before per-case z-score. Pass '--no-hu-clip' to disable.",
    )
    parser.add_argument(
        "--no-hu-clip",
        action="store_true",
        help="Disable HU clipping (already-clipped data still passes through unchanged).",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    hu_clip: Optional[Tuple[float, float]] = None if args.no_hu_clip else tuple(args.hu_clip)
    preprocess = Dataset101PMPreprocess(
        dataset_root=args.dataset_root,
        save_dir=args.save_dir,
        target_shape=tuple(args.target_shape),
        target_spacing=tuple(args.target_spacing),
        hu_clip=hu_clip,
    )
    preprocess()
