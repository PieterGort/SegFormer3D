import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def create_pandas_df(data_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(data=data_dict, index=None, columns=None)


def save_pandas_df(dataframe: pd.DataFrame, save_path: str, header: list) -> None:
    assert save_path.endswith("csv")
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe.columns) == len(header)
    dataframe.to_csv(path_or_buf=save_path, header=header, index=False)


def create_train_val_kfold_csv_from_data_folder(
    folder_dir: str,
    append_dir: str = "",
    save_dir: str = "./",
    n_k_fold: int = 5,
    random_state: int = 42,
) -> None:
    assert os.path.exists(folder_dir), f"{folder_dir} does not exist"

    header = ["data_path", "case_name"]
    case_name = next(os.walk(folder_dir), (None, None, []))[1]
    case_name = np.array(case_name)
    np.random.seed(random_state)
    np.random.shuffle(case_name)

    kfold = KFold(n_splits=n_k_fold, random_state=random_state, shuffle=True)
    os.makedirs(save_dir, exist_ok=True)

    for i, (train_fold_id, validation_fold_id) in enumerate(kfold.split(case_name)):
        train_fold_cn = case_name[train_fold_id]
        valid_fold_cn = case_name[validation_fold_id]
        train_dp = [os.path.join(append_dir, case).replace("\\", "/") for case in train_fold_cn]
        valid_dp = [os.path.join(append_dir, case).replace("\\", "/") for case in valid_fold_cn]

        train_data = {"data_path": train_dp, "case_name": train_fold_cn}
        valid_data = {"data_path": valid_dp, "case_name": valid_fold_cn}

        train_df = create_pandas_df(train_data)
        valid_df = create_pandas_df(valid_data)

        save_pandas_df(
            dataframe=train_df,
            save_path=os.path.join(save_dir, f"train_fold_{i+1}.csv"),
            header=header,
        )
        save_pandas_df(
            dataframe=valid_df,
            save_path=os.path.join(save_dir, f"validation_fold_{i+1}.csv"),
            header=header,
        )


def build_argparser() -> argparse.ArgumentParser:
    default_preprocessed_root = "/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed"
    parser = argparse.ArgumentParser(description="Create 5-fold CSV splits for Dataset101_PM.")
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
        help="Directory where fold CSV files will be written.",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    create_train_val_kfold_csv_from_data_folder(
        folder_dir=args.folder_dir,
        append_dir=args.append_dir,
        save_dir=args.save_dir,
        n_k_fold=5,
        random_state=42,
    )
