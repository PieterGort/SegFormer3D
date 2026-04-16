import argparse
import os
import random

import numpy as np
import pandas as pd


def create_train_val_test_csv_from_data_folder(
    folder_dir: str,
    append_dir: str = "",
    save_dir: str = "./",
    train_split_perc: float = 0.80,
    val_split_perc: float = 0.05,
) -> None:
    assert os.path.exists(folder_dir), f"{folder_dir} does not exist"
    assert 0.0 < train_split_perc < 1.0, "train split should be between 0 and 1"
    assert 0.0 < val_split_perc < 1.0, "validation split should be between 0 and 1"

    np.random.seed(42)
    random.seed(42)

    case_name = next(os.walk(folder_dir), (None, None, []))[1]
    corpus_sample_count = len(case_name)

    data_dir = [os.path.join(append_dir, case).replace("\\", "/") for case in case_name]

    idx = np.arange(0, corpus_sample_count)
    np.random.shuffle(idx)

    train_idx, val_idx, test_idx = np.split(
        idx,
        [
            int(train_split_perc * corpus_sample_count),
            int((train_split_perc + val_split_perc) * corpus_sample_count),
        ],
    )

    train_sample_base_dir = np.array(data_dir)[train_idx]
    train_sample_case_name = np.array(case_name)[train_idx]

    val_idx = np.concatenate((val_idx, test_idx), axis=0)
    validation_sample_base_dir = np.array(data_dir)[val_idx]
    validation_sample_case_name = np.array(case_name)[val_idx]

    train_df = pd.DataFrame(
        data={"base_dir": train_sample_base_dir, "case_name": train_sample_case_name},
    )
    validation_df = pd.DataFrame(
        data={
            "base_dir": validation_sample_base_dir,
            "case_name": validation_sample_case_name,
        },
    )

    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(
        os.path.join(save_dir, "train.csv"),
        header=["data_path", "case_name"],
        index=False,
    )
    validation_df.to_csv(
        os.path.join(save_dir, "validation.csv"),
        header=["data_path", "case_name"],
        index=False,
    )


def build_argparser() -> argparse.ArgumentParser:
    default_preprocessed_root = "/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed"
    parser = argparse.ArgumentParser(description="Create default train/validation CSVs for Dataset101_PM.")
    parser.add_argument(
        "--folder-dir",
        type=str,
        default=default_preprocessed_root,
        help="Path to the preprocessed case directory.",
    )
    parser.add_argument(
        "--append-dir",
        type=str,
        default=default_preprocessed_root,
        help="Absolute case prefix stored in each CSV row.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=default_preprocessed_root,
        help="Directory where train.csv and validation.csv will be written.",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    create_train_val_test_csv_from_data_folder(
        folder_dir=args.folder_dir,
        append_dir=args.append_dir,
        save_dir=args.save_dir,
        train_split_perc=0.80,
        val_split_perc=0.05,
    )
