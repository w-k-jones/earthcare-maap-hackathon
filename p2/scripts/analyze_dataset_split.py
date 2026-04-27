"""
Analyze EarthCARE patch distributions for a deterministic train/val/test split.

The script reports input statistics, train-normalized input shift, target count
statistics, per-patch target summaries, and invalid patch files.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import xarray as xr

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

CATEGORICAL_INPUTS = {
    "multiple_scattering_status",
    "simplified_convective_classification",
}


def make_filelist(dataset_path):
    return sorted(str(f) for f in Path(dataset_path).iterdir() if f.suffix == ".h5")


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


def filter_valid_files(filepaths, required_vars):
    valid_files = []
    invalid_files = []
    for file in filepaths:
        try:
            ds = xr.open_dataset(file)
            try:
                missing = [var for var in required_vars if var not in ds]
                if missing:
                    invalid_files.append({"path": file, "reason": f"missing: {', '.join(missing)}"})
                else:
                    valid_files.append(file)
            finally:
                ds.close()
        except Exception as exc:
            invalid_files.append({"path": file, "reason": f"open_error: {type(exc).__name__}: {exc}"})
    return valid_files, invalid_files


def empty_scalar_stats():
    return {
        "count": 0,
        "sum": 0.0,
        "sq_sum": 0.0,
        "min": np.inf,
        "max": -np.inf,
        "nonzero_count": 0,
        "nan_count": 0,
    }


def update_scalar_stats(stats, values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(arr)
    stats["nan_count"] += int((~finite).sum())
    arr = arr[finite]
    if arr.size == 0:
        return
    stats["count"] += int(arr.size)
    stats["sum"] += float(arr.sum())
    stats["sq_sum"] += float(np.square(arr).sum())
    stats["min"] = min(stats["min"], float(arr.min()))
    stats["max"] = max(stats["max"], float(arr.max()))
    stats["nonzero_count"] += int(np.count_nonzero(arr))


def finalize_scalar_stats(stats, sample_values=None):
    count = max(stats["count"], 1)
    mean = stats["sum"] / count
    var = max(stats["sq_sum"] / count - mean**2, 0.0)
    out = {
        "count": int(stats["count"]),
        "nan_count": int(stats["nan_count"]),
        "nonzero_count": int(stats["nonzero_count"]),
        "nonzero_fraction": float(stats["nonzero_count"] / count),
        "mean": float(mean),
        "std": float(np.sqrt(var)),
        "min": None if stats["count"] == 0 else float(stats["min"]),
        "max": None if stats["count"] == 0 else float(stats["max"]),
    }
    if sample_values is not None and len(sample_values) > 0:
        arr = np.asarray(sample_values, dtype=np.float64)
        out.update(
            {
                "p01": float(np.percentile(arr, 1)),
                "p05": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        )
    return out


def sample_values(values, max_samples, rng):
    arr = np.asarray(values, dtype=np.float32).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.empty((0,), dtype=np.float32)
    if arr.size <= max_samples:
        return arr
    idx = rng.choice(arr.size, size=max_samples, replace=False)
    return arr[idx]


def compute_train_input_stats(train_files, input_vars):
    stats = {var: empty_scalar_stats() for var in input_vars if var not in CATEGORICAL_INPUTS}
    for i, file in enumerate(train_files, 1):
        if i % 250 == 0:
            print(f"Train stats pass: {i}/{len(train_files)}", flush=True)
        ds = xr.open_dataset(file)
        try:
            for var in stats:
                update_scalar_stats(stats[var], ds[var].values)
        finally:
            ds.close()

    out = {}
    for var, values in stats.items():
        finalized = finalize_scalar_stats(values)
        out[var] = {"mean": finalized["mean"], "std": max(finalized["std"], 1e-6)}
    return out


def analyze_split_files(split_name, files, input_vars, target_vars, train_input_stats, max_input_samples, seed):
    rng = np.random.default_rng(seed)
    input_stats = {var: empty_scalar_stats() for var in input_vars}
    input_z_stats = {var: empty_scalar_stats() for var in input_vars if var not in CATEGORICAL_INPUTS}
    input_samples = {var: [] for var in input_vars}
    input_z_samples = {var: [] for var in input_z_stats}
    categorical_counts = {var: {} for var in input_vars if var in CATEGORICAL_INPUTS}

    target_stats = {var: empty_scalar_stats() for var in target_vars}
    target_log_stats = {var: empty_scalar_stats() for var in target_vars}
    target_samples = {var: [] for var in target_vars}
    target_patch_rows = []

    z_threshold_counts = {
        var: {"abs_z_gt_3": 0, "abs_z_gt_5": 0, "abs_z_gt_10": 0, "count": 0}
        for var in input_z_stats
    }

    for i, file in enumerate(files, 1):
        if i % 250 == 0:
            print(f"{split_name} analysis pass: {i}/{len(files)}", flush=True)

        ds = xr.open_dataset(file)
        try:
            for var in input_vars:
                arr = ds[var].values.astype(np.float32)
                update_scalar_stats(input_stats[var], arr)
                input_samples[var].extend(sample_values(arr, max_input_samples // max(len(files), 1), rng))

                if var in CATEGORICAL_INPUTS:
                    vals = arr[np.isfinite(arr)]
                    unique, counts = np.unique(vals, return_counts=True)
                    for value, count in zip(unique, counts):
                        key = str(float(value))
                        categorical_counts[var][key] = categorical_counts[var].get(key, 0) + int(count)
                    continue

                mean = train_input_stats[var]["mean"]
                std = train_input_stats[var]["std"]
                z = (arr - mean) / std
                update_scalar_stats(input_z_stats[var], z)
                input_z_samples[var].extend(sample_values(z, max_input_samples // max(len(files), 1), rng))

                z_finite = z[np.isfinite(z)]
                z_threshold_counts[var]["count"] += int(z_finite.size)
                z_threshold_counts[var]["abs_z_gt_3"] += int((np.abs(z_finite) > 3).sum())
                z_threshold_counts[var]["abs_z_gt_5"] += int((np.abs(z_finite) > 5).sum())
                z_threshold_counts[var]["abs_z_gt_10"] += int((np.abs(z_finite) > 10).sum())

            patch_row = {"split": split_name, "file": Path(file).name}
            for var in target_vars:
                y = ds[var].values.astype(np.float32)
                y = np.nan_to_num(y, nan=0.0)
                y_log = np.log1p(np.clip(y, a_min=0.0, a_max=None))

                update_scalar_stats(target_stats[var], y)
                update_scalar_stats(target_log_stats[var], y_log)
                target_samples[var].extend(y[np.isfinite(y)])

                finite = y[np.isfinite(y)]
                patch_row[f"{var}_sum"] = float(finite.sum()) if finite.size else 0.0
                patch_row[f"{var}_max"] = float(finite.max()) if finite.size else 0.0
                patch_row[f"{var}_mean"] = float(finite.mean()) if finite.size else 0.0
                patch_row[f"{var}_nonzero_fraction"] = float(np.count_nonzero(finite) / max(finite.size, 1))
            target_patch_rows.append(patch_row)
        finally:
            ds.close()

    input_summary = {}
    for var in input_vars:
        input_summary[var] = finalize_scalar_stats(input_stats[var], input_samples[var])
        if var in categorical_counts:
            input_summary[var]["categorical_counts"] = categorical_counts[var]

    input_z_summary = {}
    for var in input_z_stats:
        input_z_summary[var] = finalize_scalar_stats(input_z_stats[var], input_z_samples[var])
        z_count = max(z_threshold_counts[var]["count"], 1)
        input_z_summary[var]["abs_z_gt_3_fraction"] = z_threshold_counts[var]["abs_z_gt_3"] / z_count
        input_z_summary[var]["abs_z_gt_5_fraction"] = z_threshold_counts[var]["abs_z_gt_5"] / z_count
        input_z_summary[var]["abs_z_gt_10_fraction"] = z_threshold_counts[var]["abs_z_gt_10"] / z_count

    target_summary = {
        var: finalize_scalar_stats(target_stats[var], target_samples[var])
        for var in target_vars
    }
    target_log_summary = {
        var: finalize_scalar_stats(target_log_stats[var])
        for var in target_vars
    }

    return {
        "num_files": len(files),
        "input_raw": input_summary,
        "input_normalized_with_train_stats": input_z_summary,
        "target_raw": target_summary,
        "target_log1p": target_log_summary,
        "target_patch_rows": target_patch_rows,
    }


def write_target_patch_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary_rows(summary, group_name):
    rows = []
    for split, split_summary in summary.items():
        for var, stats in split_summary.items():
            row = {"split": split, "group": group_name, "variable": var}
            for key, value in stats.items():
                if isinstance(value, dict):
                    continue
                row[key] = value
            rows.append(row)
    return rows


def write_summary_csv(rows, path):
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = ["split", "group", "variable"] + sorted(keys - {"split", "group", "variable"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=257)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-input-samples", type=int, default=200_000)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "dataset_analysis"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_name = (
        f"analysis_seed{args.seed}_"
        f"train{args.train_ratio:g}_"
        f"val{args.val_ratio:g}_"
        f"test{args.test_ratio:g}"
    )

    print("Listing patch files...", flush=True)
    all_files = make_filelist(args.data_dir)
    print(f"Found {len(all_files)} .h5 files", flush=True)

    print("Filtering valid files...", flush=True)
    valid_files, invalid_files = filter_valid_files(all_files, INPUT_VARS + TARGET_VARS)
    invalid_path = output_dir / f"{analysis_name}_invalid_files.txt"
    with open(invalid_path, "w", encoding="utf-8") as f:
        for item in invalid_files:
            f.write(f"{item['path']}\t{item['reason']}\n")
    print(f"Valid files: {len(valid_files)} | invalid files: {len(invalid_files)}", flush=True)

    splits = split_filelist(valid_files, args.train_ratio, args.val_ratio, args.seed)
    print(
        "Split sizes | "
        f"train: {len(splits['train'])} | "
        f"val: {len(splits['val'])} | "
        f"test: {len(splits['test'])}",
        flush=True,
    )

    print("Computing train input normalization stats...", flush=True)
    train_input_stats = compute_train_input_stats(splits["train"], INPUT_VARS)

    split_results = {}
    patch_rows = []
    for split_name, files in splits.items():
        print(f"Analyzing {split_name}...", flush=True)
        result = analyze_split_files(
            split_name=split_name,
            files=files,
            input_vars=INPUT_VARS,
            target_vars=TARGET_VARS,
            train_input_stats=train_input_stats,
            max_input_samples=args.max_input_samples,
            seed=args.seed,
        )
        patch_rows.extend(result.pop("target_patch_rows"))
        split_results[split_name] = result

    output = {
        "config": {
            "data_dir": args.data_dir,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "num_all_files": len(all_files),
            "num_valid_files": len(valid_files),
            "num_invalid_files": len(invalid_files),
            "invalid_files_path": str(invalid_path),
            "input_vars": INPUT_VARS,
            "target_vars": TARGET_VARS,
        },
        "train_input_normalization_stats": train_input_stats,
        "splits": split_results,
    }

    json_path = output_dir / f"{analysis_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    rows = []
    rows += flatten_summary_rows({k: v["input_raw"] for k, v in split_results.items()}, "input_raw")
    rows += flatten_summary_rows(
        {k: v["input_normalized_with_train_stats"] for k, v in split_results.items()},
        "input_normalized_with_train_stats",
    )
    rows += flatten_summary_rows({k: v["target_raw"] for k, v in split_results.items()}, "target_raw")
    rows += flatten_summary_rows({k: v["target_log1p"] for k, v in split_results.items()}, "target_log1p")

    summary_csv_path = output_dir / f"{analysis_name}_summary.csv"
    patch_csv_path = output_dir / f"{analysis_name}_target_patch_stats.csv"
    write_summary_csv(rows, summary_csv_path)
    write_target_patch_csv(patch_rows, patch_csv_path)

    print(f"Analysis JSON saved: {json_path}", flush=True)
    print(f"Summary CSV saved: {summary_csv_path}", flush=True)
    print(f"Target patch CSV saved: {patch_csv_path}", flush=True)
    print(f"Invalid files saved: {invalid_path}", flush=True)


if __name__ == "__main__":
    main()
