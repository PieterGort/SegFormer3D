import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

root = "/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed_v4_anis_native"
case = "00033"  # change this
hu_min, hu_max = -175, 250

image = torch.load(os.path.join(root, case, f"{case}_modalities.pt"), map_location="cpu")
label = torch.load(os.path.join(root, case, f"{case}_label.pt"), map_location="cpu")

image = image[0].numpy()          # (W, H, D)
label = label[0].numpy().astype(int)

with open(os.path.join(root, "meta.json"), "r", encoding="utf-8") as infile:
    meta = json.load(infile)

ct_stats = meta["intensity_stats"]["0000"]
image_hu = image * ct_stats["std"] + ct_stats["mean"]
image_hu = np.clip(image_hu, hu_min, hu_max)

planes = {
    "sagittal": {
        "slice_axis": 0,
        "sum_axes": (1, 2),
        "slice_fn": lambda arr, idx: arr[idx, :, :],
    },
    "coronal": {
        "slice_axis": 1,
        "sum_axes": (0, 2),
        "slice_fn": lambda arr, idx: arr[:, idx, :],
    },
    "axial": {
        "slice_axis": 2,
        "sum_axes": (0, 1),
        "slice_fn": lambda arr, idx: arr[:, :, idx],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (plane_name, plane) in zip(axes, planes.items()):
    # Pick the slice with the most foreground voxels for this anatomical plane.
    fg_per_slice = (label > 0).sum(axis=plane["sum_axes"])
    slice_idx = int(np.argmax(fg_per_slice))

    img_slice = plane["slice_fn"](image_hu, slice_idx)
    lab_slice = plane["slice_fn"](label, slice_idx)

    masked_label = np.ma.masked_where(lab_slice == 0, lab_slice)
    display_img = np.rot90(img_slice, k=2)
    display_label = np.rot90(masked_label, k=2)

    ax.imshow(display_img, cmap="gray", vmin=hu_min, vmax=hu_max)
    ax.imshow(display_label, cmap="tab20", alpha=0.45, interpolation="nearest")
    ax.set_title(
        f"{plane_name} | axis {plane['slice_axis']}={slice_idx}\n"
        f"foreground voxels={fg_per_slice[slice_idx]}"
    )
    ax.axis("off")

fig.suptitle(f"{case} preprocessed image + label overlay ({hu_min} to {hu_max} HU)")
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), f"{case}_preprocessed_planes.png")
plt.savefig(output_path, dpi=200)
print(f"saved {output_path}")
plt.show()