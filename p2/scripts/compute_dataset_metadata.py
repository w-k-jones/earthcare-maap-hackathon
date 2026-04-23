"""
Create deterministic EarthCARE dataset split files and cached train statistics.

The generated JSON files can be passed to training scripts so normalization
statistics do not need to be recomputed for every run.
"""

import argparse
import json
import random
from pathlib import Path

import xarray as xr

from datamodule import compute_input_stats, make_filelist, save_splits

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = "/shared/home/ggoracci/Data/EarthCARE/patches"

INPUT_VARS = [
    "ice_water_content",
    "ice_mass_flux",
    "ice_effective_radius",
    "ice_median_volume_diameter",
    "ice_riming_factor",
    "rain_rate",
    "rain_water_content",
    "rain_median_volume_diameter",
    "liquid_water_content",
    "liquid_number_concentration",
    "liquid_effective_radius",
    "aerosol_number_concentration",
    "aerosol_mass_content",
    "doppler_velocity_best_estimate",
    "sedimentation_velocity_best_estimate",
    "spectrum_width_integrated",
    "reflectivity_no_attenuation_correction",
    "reflectivity_corrected",
    "multiple_scattering_status",
    "simplified_convective_classification",
]

TARGET_VARS = [
    "lightning_count_2p5",
    "lightning_count_5",
]


def filter_valid_files(filepaths, required_vars):
    valid_files = []
    invalid_files = []

    for file in filepaths:
        try:
            ds = xr.open_dataset(file)
            try:
                missing = [var for var in required_vars if var not in ds]
                if missing:
                    invalid_files.append(
                        {
                            "path": file,
                            "reason": f"missing variables: {', '.join(missing)}",
                        }
                    )
                else:
                    valid_files.append(file)
            finally:
                ds.close()
        except Exception as exc:
            invalid_files.append(
                {
                    "path": file,
                    "reason": f"open error: {type(exc).__name__}: {exc}",
                }
            )

    return valid_files, invalid_files


def split_filelist(files, train_ratio, val_ratio, seed):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "dataset_metadata"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = (
        f"split_seed{args.seed}_"
        f"train{args.train_ratio:g}_"
        f"val{args.val_ratio:g}_"
        f"test{args.test_ratio:g}"
    )
    splits_path = output_dir / f"{split_name}.json"
    stats_path = output_dir / f"{split_name}_train_input_stats.json"
    invalid_path = output_dir / f"{split_name}_invalid_files.txt"

    print("Listing patch files...", flush=True)
    all_files = make_filelist(args.data_dir)
    print(f"Found {len(all_files)} .h5 files", flush=True)

    print("Filtering files with missing variables...", flush=True)
    valid_files, invalid_files = filter_valid_files(
        all_files,
        required_vars=INPUT_VARS + TARGET_VARS,
    )
    with open(invalid_path, "w", encoding="utf-8") as f:
        for item in invalid_files:
            f.write(f"{item['path']}\t{item['reason']}\n")
    print(
        "Validation complete | "
        f"valid: {len(valid_files)} | "
        f"invalid: {len(invalid_files)} | "
        f"invalid list: {invalid_path}",
        flush=True,
    )

    print("Creating deterministic split...", flush=True)
    splits = split_filelist(
        files=valid_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    save_splits(splits, splits_path)
    print(
        "Split saved | "
        f"train: {len(splits['train'])} | "
        f"val: {len(splits['val'])} | "
        f"test: {len(splits['test'])} | "
        f"path: {splits_path}",
        flush=True,
    )

    print("Computing train input statistics...", flush=True)
    stats = compute_input_stats(
        filepaths=splits["train"],
        input_vars=INPUT_VARS,
        output_path=stats_path,
    )
    print(f"Stats saved: {stats_path}", flush=True)

    manifest = {
        "data_dir": args.data_dir,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "splits_path": str(splits_path),
        "stats_path": str(stats_path),
        "input_vars": INPUT_VARS,
        "target_vars": TARGET_VARS,
        "invalid_files_path": str(invalid_path),
        "num_all_files": len(all_files),
        "num_valid_files": len(valid_files),
        "num_invalid_files": len(invalid_files),
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
    }
    manifest_path = output_dir / f"{split_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
