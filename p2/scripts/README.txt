EarthCARE training scripts
==========================

Run commands should be executed from this directory:

  cd /shared/home/ggoracci/Data/EarthCARE/earthcare-maap-hackathon/p2/scripts


1. Full training with main.py
----------------------------

main.py runs the full training configuration using UNetSkip.
It writes artifacts to:

  ../runs/<run-name>/

including:

  training_images/
  model.pt
  history.pkl

Basic run:

  python main.py --run-name full_unetskip_run

Run with precomputed split and normalization statistics:

  python main.py \
    --run-name full_unetskip_cached \
    --splits-path ../dataset_metadata/split_seed257_train0.7_val0.2_test0.1.json \
    --stats-path ../dataset_metadata/split_seed257_train0.7_val0.2_test0.1_train_input_stats.json


2. Subset ProfileCNN training
-----------------------------

run_subset_profile_cnn.py runs a smaller ProfileCNN experiment. It is useful for
debugging, testing variable subsets, and faster model/loss experiments.

Basic run with default subset sizes:

  python run_subset_profile_cnn.py --run-name profile_subset_debug

Custom subset sizes:

  python run_subset_profile_cnn.py \
    --run-name profile_subset_1000 \
    --max-train-patches 1000 \
    --max-val-patches 200 \
    --max-test-patches 200

Run with precomputed split and normalization statistics:

  python run_subset_profile_cnn.py \
    --run-name profile_subset_cached \
    --splits-path ../dataset_metadata/split_seed257_train0.7_val0.2_test0.1.json \
    --stats-path ../dataset_metadata/split_seed257_train0.7_val0.2_test0.1_train_input_stats.json \
    --max-train-patches 1000 \
    --max-val-patches 200


3. Precompute split and normalization statistics
------------------------------------------------

This creates deterministic train/val/test split files and train-set input
normalization statistics:

  python compute_dataset_metadata.py --seed 257

Outputs are written to:

  ../dataset_metadata/


4. Analyze dataset distribution
-------------------------------

This computes input and target statistics for the deterministic split:

  python analyze_dataset_split.py --seed 257

Outputs are written to:

  ../dataset_analysis/


Notes
-----

- Generated folders ../runs, ../dataset_metadata, and ../dataset_analysis are
  ignored by git.
- Targets are trained with log1p transformation when target_log1p=True.
- Saved prediction plots are converted back to count scale for readability.
