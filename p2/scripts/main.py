"""
Run full EarthCARE lightning-count training with the UNetSkip model.

This script trains on the configured patch directory, writes run artifacts under
p2/runs/<run-name>/, and can optionally reuse precomputed split/statistics JSON
files.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import torch

from datamodule import EarthCARELightningDataModule
from models.unetskip import UNetSkip
from models.profile_cnn import ProfileCNN
from train import train

PROJECT_DIR = Path(__file__).resolve().parents[1]
data_dir = "/shared/home/ggoracci/Data/EarthCARE/patches"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-name",
    default=None,
    help="Optional run folder name. If omitted, a timestamp is used.",
)
parser.add_argument("--splits-path", default=None)
parser.add_argument("--stats-path", default=None)
parser.add_argument("--split-seed", type=int, default=42)
args = parser.parse_args()

run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = PROJECT_DIR / "runs" / run_name
if run_dir.exists():
    suffix = datetime.now().strftime("%H%M%S")
    run_dir = PROJECT_DIR / "runs" / f"{run_name}_{suffix}"

training_images_dir = run_dir / "training_images"
training_images_dir.mkdir(parents=True, exist_ok=False)

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
device = "cuda" if cuda_available else "cpu"
print(f"Selected training device: {device}")

input_vars = [
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

target_vars = [
    "lightning_count_2p5",
    #"lightning_count_5",
]
print(f"Run directory: {run_dir}")
print("Creating datamodule")
dm = EarthCARELightningDataModule(
    data_dir=data_dir,
    input_vars=input_vars,
    target_vars=target_vars,
    batch_size=128,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    fill_value=0.0,
    norm_with_train=True,
    target_log1p=True,
    split_seed=args.split_seed,
    splits_path=args.splits_path,
    stats_path=args.stats_path,
)
model = ProfileCNN(
    in_channels=len(input_vars),
    out_channels=len(target_vars),
    base_channels=32,
    nonnegative_output=False,
)
print(f"Model initial device: {next(model.parameters()).device}")
print("Starting Training")
history = train(
    model=model,
    datamodule=dm,
    epochs=20,
    lr=1e-4,
    device=device,
    criterion=torch.nn.MSELoss(),
    plot=True,
    plot_every=1,
    plot_sample_idx=0,
    plot_target_channel=0,
    plot_save_dir=str(training_images_dir),
    show_plot=False,
)
print("Training Complete")
torch.save(model.state_dict(), run_dir / "model.pt")
with open(run_dir / "history.pkl", "wb") as f:
    pickle.dump(history, f)
print(f"Model saved: {run_dir / 'model.pt'}")
print(f"History saved: {run_dir / 'history.pkl'}")
