import sys

sys.path.append("../")

from typing import Dict
from monai.data import DataLoader
from augmentations.augmentations import build_augmentations


######################################################################
def build_dataset(dataset_type: str, dataset_args: Dict):
    transform = build_augmentations(
        train=dataset_args["train"],
        dataset_type=dataset_type,
        roi_size=dataset_args.get("roi_size", (96, 96, 96)),
        num_samples=dataset_args.get("num_samples", 4),
    )

    if dataset_type == "brats2021_seg":
        from .brats2021_seg import Brats2021Task1Dataset

        dataset = Brats2021Task1Dataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=transform,
            fold_id=dataset_args["fold_id"],
        )
        return dataset
    elif dataset_type == "brats2017_seg":
        from .brats2017_seg import Brats2017Task1Dataset

        dataset = Brats2017Task1Dataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=transform,
            fold_id=dataset_args["fold_id"],
        )
        return dataset
    elif dataset_type == "nnunet_seg":
        from .nnunet_seg import NnUNetSegDataset

        dataset = NnUNetSegDataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            transform=transform,
            fold_id=dataset_args.get("fold_id"),
            val_fraction=dataset_args.get("val_fraction", 0.2),
            split_seed=dataset_args.get("split_seed", 42),
            image_dir=dataset_args.get("image_dir", "imagesTr"),
            label_dir=dataset_args.get("label_dir", "labelsTr"),
            dataset_json=dataset_args.get("dataset_json", "dataset.json"),
            clip_percentiles=dataset_args.get("clip_percentiles", (0.5, 99.5)),
        )
        return dataset
    else:
        raise ValueError(
            "only brats2021, brats2017, and nnunet_seg segmentation are currently supported!"
        )


######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict, config: Dict = None, train: bool = True
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    num_workers = dataloader_args["num_workers"]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=num_workers,
        drop_last=dataloader_args["drop_last"],
        pin_memory=True,
        # Keep worker processes alive between epochs to avoid the ~1-2s
        # spawn overhead at the start of every epoch (only meaningful when
        # num_workers > 0).
        persistent_workers=num_workers > 0,
        prefetch_factor=dataloader_args.get("prefetch_factor", 2) if num_workers > 0 else None,
    )
    return dataloader
