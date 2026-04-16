import monai.transforms as transforms
from typing import Any, Iterable, Tuple


def _to_tuple3(roi_size: Iterable[int]) -> Tuple[int, int, int]:
    roi = tuple(int(x) for x in roi_size)
    if len(roi) != 3:
        raise ValueError(f"roi_size must contain exactly 3 values, got {roi_size}")
    return roi

#######################################################################################
def build_augmentations(
    train: bool = True,
    dataset_type: str = "brats2017_seg",
    roi_size=(96, 96, 96),
    num_samples: int = 4,
) -> transforms.Compose:
    """Build data augmentation pipeline for 3D medical image segmentation.
    
    Training augmentations include:
    - Random spatial cropping with 4 samples per volume
    - Random horizontal flipping (30% probability)
    - Random rotation around x-axis (50% probability, ±20.6°)
    - Coarse dropout for regularization (50% probability)
    - Gibbs noise to simulate MRI artifacts
    
    Validation uses minimal transforms (type conversion only).
    
    Args:
        train: If True, returns training augmentations. If False, returns validation transforms.
        
    Returns:
        Composed MONAI transform pipeline
    """
    roi_size = _to_tuple3(roi_size)

    if dataset_type == "nnunet_seg":
        if train:
            train_transform = [
                # Focus patch sampling on labeled anatomy instead of the large CT background.
                transforms.CropForegroundd(
                    keys=["image", "label"],
                    source_key="label",
                ),
                transforms.SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=roi_size,
                ),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi_size,
                    pos=1,
                    neg=1,
                    num_samples=num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"],
                    prob=0.50,
                    spatial_axis=0,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"],
                    prob=0.50,
                    spatial_axis=1,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"],
                    prob=0.50,
                    spatial_axis=2,
                ),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=0.50,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(
                    keys=["image"],
                    factors=0.10,
                    prob=0.50,
                ),
                transforms.RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                transforms.EnsureTyped(
                    keys=["image", "label"],
                    track_meta=False,
                ),
            ]
            return transforms.Compose(train_transform)

        val_transform = [
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
            ),
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=roi_size,
            ),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=roi_size,
            ),
            transforms.EnsureTyped(
                keys=["image", "label"],
                track_meta=False,
            ),
        ]
        return transforms.Compose(val_transform)

    if train:
        train_transform = [
            # Random spatial cropping - generates 4 crops per volume for data efficiency
            transforms.RandSpatialCropSamplesd(
                keys=["image", "label"], 
                roi_size=(96, 96, 96), 
                num_samples=4, 
                random_center=True, 
                random_size=False
            ),
            # Random horizontal flip for geometric augmentation
            transforms.RandFlipd(
                keys=["image", "label"], 
                prob=0.30, 
                spatial_axis=1
            ),
            # Random rotation around x-axis (sagittal plane)
            transforms.RandRotated(
                keys=["image", "label"], 
                prob=0.50, 
                range_x=0.36,  # ±20.6 degrees
                range_y=0.0, 
                range_z=0.0
            ),
            # Coarse dropout for robustness
            transforms.RandCoarseDropoutd(
                keys=["image", "label"], 
                holes=20, 
                spatial_size=(-1, 7, 7), 
                fill_value=0, 
                prob=0.5
            ),
            # Gibbs ringing artifact simulation (MRI-specific)
            transforms.GibbsNoised(keys=["image"]),
            # Ensure proper tensor types without metadata tracking (faster)
            transforms.EnsureTyped(
                keys=["image", "label"], 
                track_meta=False
            ),
        ]
        return transforms.Compose(train_transform)
    else:
        # Minimal validation transforms - only ensure type consistency
        val_transform = [
            transforms.EnsureTyped(
                keys=["image", "label"], 
                track_meta=False
            ),
        ]
        return transforms.Compose(val_transform)
