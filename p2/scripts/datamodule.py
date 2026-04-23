from torch.utils.data import DataLoader
import json
from pathlib import Path
import random
import xarray as xr
import numpy as np
from dataset import EarthCARELightningDataset


def load_input_stats(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_input_stats(stats, stats_path):
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def compute_input_stats(filepaths, input_vars, output_path=None):
    output_dict = {
        var: {
            "sum": 0.0,
            "counts": 0,
            "sq_sum": 0.0,
        }
        for var in input_vars
    }

    for file in filepaths:
        ds = read_one_patch(file)
        try:
            for var in input_vars:
                if var not in ds:
                    raise KeyError(f"Missing variable {var!r} in file {file}")
                arr = ds[var].values.astype(np.float32)
                mask = np.isfinite(arr)
                valids = arr[mask]

                if valids.size == 0:
                    continue

                output_dict[var]["sum"] += float(valids.sum())
                output_dict[var]["counts"] += int(valids.size)
                output_dict[var]["sq_sum"] += float(np.square(valids).sum())

        finally:
            if hasattr(ds, "close"):
                ds.close()

    for values in output_dict.values():
        values["mean"] = values["sum"] / max(values["counts"], 1)
        var = values["sq_sum"] / max(values["counts"], 1) - values["mean"] ** 2
        values["std"] = float(np.sqrt(max(var, 1e-6)))

        values["sum"] = float(values["sum"])
        values["sq_sum"] = float(values["sq_sum"])
        values["counts"] = int(values["counts"])
        values["mean"] = float(values["mean"])

    if output_path is not None:
        save_input_stats(output_dict, output_path)

    return output_dict   

def make_filelist(dataset_path):
    dataset_path = Path(dataset_path)
    return sorted(str(f) for f in dataset_path.iterdir() if f.suffix == ".h5")

def read_one_patch(file):
    return xr.open_dataset(file)
    
def random_split_dataset(
    dataset_dir,
    train_ratio=0.7,
    val_ratio=0.20,
    test_ratio=0.10,
    seed=42):

    dataset_dir = Path(dataset_dir)

    files = make_filelist(dataset_dir)

    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    split_dict = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    return split_dict


def save_splits(splits_dict, splits_path):
    splits_path = Path(splits_path)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2)


def load_splits(splits_path):
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


class EarthCARELightningDataModule:
    def __init__(
        self,
        data_dir: str,
        input_vars,
        target_vars,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        fill_value: float = 0.0,
        norm_with_train: bool = True,
        target_log1p: bool = False,
        split_seed: int = 42,
        splits_path: str | None = None,
        stats_path: str | None = None,
        persistent_workers: bool = False,
    ):
        self.data_dir = data_dir
        if splits_path is None:
            self.splits_dict = random_split_dataset(data_dir, seed=split_seed)
        else:
            self.splits_dict = load_splits(splits_path)
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fill_value = fill_value
        self.norm_with_train = norm_with_train
        self.target_log1p = target_log1p
        self.split_seed = split_seed
        self.splits_path = splits_path
        self.stats_path = stats_path
        self.persistent_workers = persistent_workers
            

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_files = self.splits_dict["train"]
        val_files = self.splits_dict["val"]
        test_files = self.splits_dict["test"]

        if self.stats_path is not None:
            mean_std_dict_train = load_input_stats(self.stats_path)
            mean_std_dict_val = mean_std_dict_train
            mean_std_dict_test = mean_std_dict_train
        elif self.norm_with_train:
            mean_std_dict_train = compute_input_stats(train_files, self.input_vars)
            mean_std_dict_val = mean_std_dict_train
            mean_std_dict_test = mean_std_dict_train
        else:
            mean_std_dict_train = compute_input_stats(train_files, self.input_vars)
            mean_std_dict_val = compute_input_stats(val_files, self.input_vars)
            mean_std_dict_test = compute_input_stats(test_files, self.input_vars)

        self.train_dataset = EarthCARELightningDataset(
            filelist=train_files,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            mean_std_dict=mean_std_dict_train,
            fill_value=self.fill_value,
            target_log1p=self.target_log1p,
        )

        self.val_dataset = EarthCARELightningDataset(
            filelist=val_files,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            mean_std_dict=mean_std_dict_val,
            fill_value=self.fill_value,
            target_log1p=self.target_log1p,
        )

        self.test_dataset = EarthCARELightningDataset(
            filelist=test_files,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            mean_std_dict=mean_std_dict_test,
            fill_value=self.fill_value,
            target_log1p=self.target_log1p,
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
