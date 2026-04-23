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
PREPROCESS_VERSION = 3


class Dataset101PMPreprocess:
    """Preprocess Dataset101_PM (nnU-Net style CT) into fast-loading .pt tensors.

    Output layout (v3):
      save_dir/
        <case>/<case>_modalities.pt   # float32 (C, W, H, D) image tensor
        <case>/<case>_label.pt        # uint8  (1, W, H, D) integer label map
        foreground_stats.json         # per-channel foreground intensity stats
        meta.json                     # preprocessing metadata + version marker

    Key differences vs v2 (which produced the 21981180 / 21983799 lr-sweep runs):
      * Image normalization switched to nnU-Net's CTNormalization:
          1. Compute per-channel foreground intensity stats ONCE across the
             whole dataset (label > 0 voxels) — clip percentiles (default
             0.5 / 99.5), mean, std.
          2. Clip each case's intensities to the global percentile window.
          3. z-score with the global mean/std.
        v2 used per-case z-score after a hard-coded HU clip to [-175, 250],
        which (a) threw away ~70% of the raw HU range (dataset still has
        min=-1024 / max=3071 per nnU-Net's analysis) and (b) let per-case
        air/background ratio bias the per-case mean.  v3 gives every case the
        same reference frame and preserves information outside the
        soft-tissue window.

    Resampling and final shape are unchanged from v2 (isotropic 2 mm,
    center-crop-or-pad to target_shape) so v2 vs v3 is a clean A/B on
    normalization only.
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
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        num_foreground_samples_per_case: int = 10_000,
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
        self.clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        self.num_foreground_samples_per_case = int(num_foreground_samples_per_case)
        self.save_dir = os.path.abspath(save_dir)

        # Populated by compute_intensity_stats() (or loaded from cache).
        # Shape: {channel_code: {"clip_lower", "clip_upper", "mean", "std", ...}}.
        self.intensity_stats: Optional[Dict[str, Dict[str, float]]] = None

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

    def normalize(self, x: np.ndarray, channel_code: str) -> np.ndarray:
        """nnU-Net-style CTNormalization using dataset-global foreground stats.

        Pipeline (per channel):
          1. Clip to the dataset's foreground percentile window
             [clip_lower, clip_upper] — same clip for every case.
          2. Standardize with the dataset-global foreground mean/std
             (computed on unclipped foreground voxels, matching nnU-Net).

        Using global (not per-case) stats gives every case the same reference
        frame, so the model doesn't have to compensate for per-case variation
        in background/air ratios.
        """
        if self.intensity_stats is None or channel_code not in self.intensity_stats:
            raise RuntimeError(
                "Intensity stats are not available. "
                "Call compute_intensity_stats() before preprocessing."
            )
        stats = self.intensity_stats[channel_code]
        x = x.astype(np.float32, copy=False)
        x = np.clip(x, stats["clip_lower"], stats["clip_upper"])
        std = max(float(stats["std"]), 1e-6)
        x = (x - float(stats["mean"])) / std
        return x.astype(np.float32, copy=False)

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

    def preprocess_modality(self, data_fp: str, channel_code: str) -> np.ndarray:
        data, affine = self.load_nifti(data_fp)
        data = self.normalize(x=data, channel_code=channel_code)
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
            modality = self.preprocess_modality(modality_fp, channel_code)
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

    # ------------------------------------------------------------------
    # Pass 1: foreground intensity stats (nnU-Net CTNormalization inputs).
    # ------------------------------------------------------------------
    def _sample_case_foreground_intensities(
        self, idx: int
    ) -> Dict[str, np.ndarray]:
        """Sample foreground-voxel intensities for one case, per channel.

        Foreground = any voxel with label > 0. For each channel we sample up
        to ``num_foreground_samples_per_case`` raw (unclipped, untransformed)
        intensities using a deterministic per-case RNG.
        """
        case_name = self.case_names[idx]
        label_fp = self.get_modality_fp(case_name, "labelsTr", None)
        label = nibabel.load(label_fp).get_fdata()
        foreground_mask = label > 0
        rng = np.random.default_rng(seed=42 + idx)

        samples: Dict[str, np.ndarray] = {}
        for channel_code in self.channel_codes:
            if not np.any(foreground_mask):
                samples[channel_code] = np.zeros(0, dtype=np.float32)
                continue
            image_fp = self.get_modality_fp(case_name, "imagesTr", channel_code)
            image = nibabel.load(image_fp).get_fdata().astype(np.float32, copy=False)
            foreground_values = image[foreground_mask]
            if foreground_values.size > self.num_foreground_samples_per_case:
                sub_idx = rng.choice(
                    foreground_values.size,
                    size=self.num_foreground_samples_per_case,
                    replace=False,
                )
                foreground_values = foreground_values[sub_idx]
            samples[channel_code] = foreground_values.astype(np.float32, copy=False)
        return samples

    def compute_intensity_stats(
        self, force: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Compute or load cached dataset-global foreground intensity stats.

        Runs Pass 1 over all cases in parallel, samples foreground voxel
        intensities (label > 0), and aggregates per-channel clip percentiles
        + mean/std. Writes the result to ``save_dir/foreground_stats.json``
        so repeat runs can reuse it.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        stats_cache_path = os.path.join(self.save_dir, "foreground_stats.json")

        if os.path.exists(stats_cache_path) and not force:
            with open(stats_cache_path, "r", encoding="utf-8") as infile:
                self.intensity_stats = json.load(infile)
            print(f"[info] loaded cached foreground intensity stats from {stats_cache_path}")
            return self.intensity_stats

        num_workers = self._resolve_num_workers()
        num_cases = len(self)
        print(
            f"[info] computing foreground intensity stats: {num_cases} cases, "
            f"up to {self.num_foreground_samples_per_case} samples per case per channel, "
            f"{num_workers} workers"
        )

        per_case_samples: List[Dict[str, np.ndarray]] = []
        with Pool(processes=num_workers) as pool:
            for sample_dict in tqdm(
                pool.imap_unordered(
                    self._sample_case_foreground_intensities, range(num_cases)
                ),
                total=num_cases,
                desc="foreground-stats",
            ):
                per_case_samples.append(sample_dict)

        p_low, p_high = self.clip_percentiles
        stats: Dict[str, Dict[str, float]] = {}
        for channel_code in self.channel_codes:
            pooled = np.concatenate(
                [d[channel_code] for d in per_case_samples if d[channel_code].size > 0]
            ) if any(d[channel_code].size > 0 for d in per_case_samples) else np.zeros(0, dtype=np.float32)
            if pooled.size == 0:
                raise RuntimeError(
                    f"No foreground voxels found for channel {channel_code}. "
                    "Every case has an all-background label?"
                )
            stats[channel_code] = {
                "clip_lower": float(np.percentile(pooled, p_low)),
                "clip_upper": float(np.percentile(pooled, p_high)),
                "mean": float(pooled.mean()),
                "std": float(max(pooled.std(), 1e-6)),
                "min": float(pooled.min()),
                "max": float(pooled.max()),
                "median": float(np.median(pooled)),
                "clip_percentiles": [float(p_low), float(p_high)],
                "num_foreground_voxels_sampled": int(pooled.size),
                "num_cases_with_foreground": int(
                    sum(1 for d in per_case_samples if d[channel_code].size > 0)
                ),
            }
            print(
                f"[info] channel {channel_code}: "
                f"clip=[{stats[channel_code]['clip_lower']:.2f}, {stats[channel_code]['clip_upper']:.2f}]  "
                f"mean={stats[channel_code]['mean']:.2f}  "
                f"std={stats[channel_code]['std']:.2f}  "
                f"(n={pooled.size})"
            )

        with open(stats_cache_path, "w", encoding="utf-8") as outfile:
            json.dump(stats, outfile, indent=2)
        print(f"[info] wrote foreground intensity stats to {stats_cache_path}")

        self.intensity_stats = stats
        return stats

    def __call__(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        if self.intensity_stats is None:
            self.compute_intensity_stats()

        num_workers = self._resolve_num_workers()
        print("started preprocessing Dataset101_PM (v%d)..." % PREPROCESS_VERSION)
        print(f"using {num_workers} worker processes")
        print(f"target spacing: {self.target_spacing} mm  |  target shape: {self.target_shape}")
        print(
            f"normalization: nnU-Net CTNormalization "
            f"(clip percentiles {self.clip_percentiles}, dataset-global z-score)  |  "
            f"label format: integer label map"
        )
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
            "normalization": "nnUNet_CTNormalization (percentile clip + global z-score)",
            "clip_percentiles": list(self.clip_percentiles),
            "intensity_stats": self.intensity_stats,
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
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=(0.5, 99.5),
        metavar=("LOW", "HIGH"),
        help=(
            "Foreground-voxel percentiles used for per-channel HU clipping. "
            "Default matches nnU-Net's CTNormalization (0.5 / 99.5)."
        ),
    )
    parser.add_argument(
        "--num-foreground-samples-per-case",
        type=int,
        default=10_000,
        help="Max foreground voxels sampled per case per channel when computing stats.",
    )
    parser.add_argument(
        "--force-recompute-stats",
        action="store_true",
        help="Recompute foreground intensity stats even if foreground_stats.json exists.",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    preprocess = Dataset101PMPreprocess(
        dataset_root=args.dataset_root,
        save_dir=args.save_dir,
        target_shape=tuple(args.target_shape),
        target_spacing=tuple(args.target_spacing),
        clip_percentiles=tuple(args.clip_percentiles),
        num_foreground_samples_per_case=args.num_foreground_samples_per_case,
    )
    preprocess.compute_intensity_stats(force=args.force_recompute_stats)
    preprocess()
