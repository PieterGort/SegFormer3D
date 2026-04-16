import argparse
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

import nibabel
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import EnsureType, Orientation
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class ConvertToMultiChannelBasedOnLabelMap:
    def __init__(self, label_values: Sequence[int]) -> None:
        self.label_values = tuple(sorted(int(label_value) for label_value in label_values))

    def __call__(self, img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [img == label_value for label_value in self.label_values]
        if isinstance(img, torch.Tensor):
            return torch.stack(result, dim=0)
        return np.stack(result, axis=0)


class Dataset101PMPreprocess:
    def __init__(
        self,
        dataset_root: str,
        save_dir: str,
        image_dir: str = "imagesTr",
        label_dir: str = "labelsTr",
        dataset_json: str = "dataset.json",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        self.dataset_root = os.path.abspath(dataset_root)
        self.image_dir = os.path.join(self.dataset_root, image_dir)
        self.label_dir = os.path.join(self.dataset_root, label_dir)
        self.dataset_meta = self._load_dataset_meta(os.path.join(self.dataset_root, dataset_json))
        self.file_ending = self.dataset_meta.get("file_ending", ".nii.gz")
        self.channel_codes = self._get_channel_codes(self.dataset_meta)
        self.label_values = self._get_label_values(self.dataset_meta)
        self.label_converter = ConvertToMultiChannelBasedOnLabelMap(self.label_values)
        self.target_shape = tuple(int(dim) for dim in target_shape)
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
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_1d_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1d_array.reshape(x.shape)
        return normalized_data.astype(np.float32, copy=False)

    @staticmethod
    def orient(x: MetaTensor) -> MetaTensor:
        return Orientation(axcodes="RAS")(x)

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
        return self.detach_meta(data)

    def preprocess_label(self, data_fp: str) -> np.ndarray:
        data, affine = self.load_nifti(data_fp)
        data = np.rint(data).astype(np.uint8, copy=False)
        data = self.label_converter(data)
        data = MetaTensor(x=data, affine=affine)
        data = self.orient(data)
        return self.detach_meta(data).astype(np.float32, copy=False)

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
        label = self._center_crop_or_pad(label)

        modalities = modalities.swapaxes(1, 3)
        label = label.swapaxes(1, 3)
        return modalities.astype(np.float32, copy=False), label.astype(np.float32, copy=False), case_name

    def __call__(self) -> None:
        num_workers = self._resolve_num_workers()
        print("started preprocessing Dataset101_PM...")
        print(f"using {num_workers} worker processes")
        with Pool(processes=num_workers) as multi_p:
            for _ in tqdm(
                multi_p.imap_unordered(self.process, range(self.__len__())),
                total=self.__len__(),
                desc="preprocess",
            ):
                pass
        print("finished preprocessing Dataset101_PM...")

    def process(self, idx: int) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        modalities, label, case_name = self.__getitem__(idx)
        data_save_path = os.path.join(self.save_dir, case_name)
        os.makedirs(data_save_path, exist_ok=True)
        torch.save(modalities, os.path.join(data_save_path, f"{case_name}_modalities.pt"))
        torch.save(label, os.path.join(data_save_path, f"{case_name}_label.pt"))
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
        default="/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed",
        help="Directory where preprocessed case folders will be saved.",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("D", "H", "W"),
        help="Final spatial shape saved for each case.",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    preprocess = Dataset101PMPreprocess(
        dataset_root=args.dataset_root,
        save_dir=args.save_dir,
        target_shape=tuple(args.target_shape),
    )
    preprocess()
