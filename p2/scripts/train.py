# ---------------------------------------------------------------------
# File overview
# Training loop and visualization utilities for downscaling models.
# ---------------------------------------------------------------------
from __future__ import annotations
import logging
import torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random
import os
from ..metrics.plotting import display_training
from ..utils.downscaling.geometry import spherical_cell_areas
from .losses import masked_mse_areaweighted, masked_wghm_mse, anti_clone_ssim
# ----------------------------
# Utilities
# ----------------------------

# def seed_everything(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # deterministic cuDNN (reproducibility > speed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# ----------------------------
# Training
# ----------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    ds_true: xr.Dataset | None,
    true_hr_var_name: str,
    ds_interp: xr.Dataset,
    coarse_var_name: str,
    interp_var_name: str,
    show_interp: bool,
    region_name: str,
    epochs: int = 50,
    lr_model: float = 1e-4,
    lambda_mse: float = 1.0,
    lambda_wghm: float = 0.01,
    #lambda_div: float = 0.0,
    lambda_ssim: float = 0.05,
    device: str = "cuda",
    WGHM: bool = False,
    plot: bool = True,
    model_name: str | None = None,
    save_flag: bool = False,
    save_path: str | None = None,
    verbose: bool = True,
) -> None:
    """
    Train a model using masked losses and optional WGHM supervision.

    Inputs:
        model: torch model.
        dataloader: DataLoader yielding dataset items.
        ds_true: optional xarray Dataset with ground truth.
        true_hr_var_name: ground-truth HR variable name.
        ds_interp: xarray Dataset with interpolated TWS.
        coarse_var_name: LR variable name.
        interp_var_name: interpolated variable name.
        show_interp: if True, show interpolated HR in diagnostics.
        region_name: region label for plot titles.
        epochs, lr_model, lambda_mse, lambda_wghm, lambda_ssim: training hyperparameters.
        device: torch device string.
        WGHM: if True, include WGHM loss.
        plot: if True, show diagnostic plots.
        model_name, save_flag, save_path: plot saving options.
        verbose: if True, log training progress.

    Outputs:
        None. (Model weights are updated in-place; optional figures saved to disk.)
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr_model)

    dset = dataloader.dataset

    # safer "fixed" index for visual diagnostics
    fixed_idx = min(550, len(dset) - 1) if len(dset) > 0 else 0

    wghm_mean, wghm_std = getattr(dset, "wghm_tws_mean", 0.0), getattr(dset, "wghm_tws_std", 1.0)

    # prepare spatial metadata once (for plots + area weights)
    # Use ds_true lat/lon restricted to HR indices used by dataset
    try:
        lat = ds_true["lat"].isel(lat=dset.lat_hr_idx).values
        lon = ds_true["lon"].isel(lon=dset.lon_hr_idx).values
    except:
        lat = ds_interp["lat"].isel(lat=dset.lat_hr_idx).values
        lon = ds_interp["lon"].isel(lon=dset.lon_hr_idx).values
        


    areas_np = spherical_cell_areas(lat, lon)  # (H,W)
    area_hr_t = torch.from_numpy(areas_np).to(device=device, dtype=torch.float32)

    for epoch in range(epochs):

        # ----------------
        # Visual snapshot
        # ----------------
        model.eval()
        with torch.no_grad():
            if len(dset) > 0:
                
                x_fix, y_fix, hydro_fix, mask3_fix, mask1_fix, mjd_fix = dset[fixed_idx]
                if hasattr(dset, "get_target_scaler"):
                    y_mean, y_std = dset.get_target_scaler(fixed_idx)
                else:
                    y_mean, y_std = dset.y_mean, dset.y_std
                # masks
                mask_hr = mask1_fix.squeeze(0).cpu().numpy().astype(bool)[0, :, :]
                mask_lr = mask3_fix.squeeze(0).cpu().numpy().astype(bool)
                

                x_fix = x_fix.unsqueeze(0).to(device)
                pred_fix = model(x_fix)

                # unnormalize pred HR
                pred_hr = pred_fix.squeeze().cpu().numpy() * y_std + y_mean

                # downsample pred to LR grid
                pred_lr_norm = F.adaptive_avg_pool2d(pred_fix, output_size=y_fix.shape[-2:])
                pred_lr = pred_lr_norm.squeeze().cpu().numpy() * y_std + y_mean

                true_lr = y_fix.squeeze().cpu().numpy() * y_std + y_mean

                t_val = mjd_fix.squeeze(0).cpu().numpy()

                pred_lr_masked = np.where(mask_lr, pred_lr, np.nan)
                true_lr_masked = np.where(mask_lr, true_lr, np.nan)
                pred_hr_masked = np.where(mask_hr, pred_hr, np.nan)
                titles = [
                    "Downscaled TWSA",
                    "Downscaled TWSA (upsampled)",
                    "Satellite TWSA",
                ]
                data_list = [
                    pred_hr_masked,
                    pred_lr_masked,
                    true_lr_masked,
                ]
                var_name = interp_var_name if interp_var_name is not None else coarse_var_name
                if true_hr_var_name is not None:
                    true_hr = ds_true[true_hr_var_name].isel(time=fixed_idx).isel(
                        lat=dset.lat_hr_idx, lon=dset.lon_hr_idx
                    ).values
                    true_hr_masked = np.where(mask_hr, true_hr, np.nan)
                    data_list.append(true_hr_masked)
                    titles.append("HISBED Reference TWSA")
                if show_interp:
                    
                    interp_hr = ds_interp[var_name].isel(time=fixed_idx).isel(
                    lat=dset.lat_hr_idx, lon=dset.lon_hr_idx
                    ).values
                    interp_hr_masked = np.where(mask_hr, interp_hr, np.nan)
                    
                    data_list.append(interp_hr_masked)
                    titles.append("Interpolated High-Res TWSA")
                if WGHM:
                    if hasattr(dset, "get_hydro_scaler"):
                        hydro_mean, hydro_std = dset.get_hydro_scaler(fixed_idx + 18)
                    else:
                        hydro_mean, hydro_std = wghm_mean, wghm_std
                    hydro_hr = hydro_fix.squeeze().cpu().numpy() * hydro_std + hydro_mean
                    hydro_hr_masked = np.where(mask_hr, hydro_hr, np.nan)
                    data_list.append(hydro_hr_masked)
                    titles.append("WGHM High-Res TWSA")

                fig, axes, images = display_training(
                    data_list,
                    titles,
                    lat=lat,
                    lon=lon,
                    cmap="coolwarm",
                    vmin=-300,
                    vmax=300,
                    cbar_label="TWSA [mm EWH]",
                    fontsize=18,
                )

                # This section can be placed where you prefer
                fig.suptitle(f"{var_name}, {region_name.capitalize()}, epoch: {epoch}", fontsize=18)

                if save_flag and save_path and model_name:
                    img_dir = os.path.join(save_path, "training")
                    os.makedirs(img_dir, exist_ok=True)
                    img_path = os.path.join(img_dir, f"{model_name}_epoch_{epoch}.png")
                    fig.savefig(img_path, dpi=150)  # better than plt.savefig

                if plot or (epoch == (epochs - 1)):
                    plt.show()

                plt.close(fig)

        model.train()

        # ----------------
        # Train loop
        # ----------------
        total_mse = 0.0
        #total_div = 0.0
        total_ssim = 0.0
        total_wghm = 0.0
        total_loss = 0.0

        for item in dataloader:
            #if WGHM:
            x, y, wghm, mask_LR, mask_HR, _ = item
            x = x.to(device)
            y = y.to(device)
            mask_LR = mask_LR.to(device)
            mask_HR = mask_HR.to(device)
            #else:
                # x, y, mask_LR, mask_HR, _ = item
                # x = x.to(device)
                # y = y.to(device)
                # mask_LR = mask_LR.to(device)
                # mask_HR = mask_HR.to(device)

            pred = model(x)
            optimizer.zero_grad(set_to_none=True)

            mse_loss = lambda_mse * masked_mse_areaweighted(
                pred_hr=pred,
                y_lr=y,
                mask_lr=mask_LR,
                mask_hr=mask_HR[0],
                area_hr=area_hr_t,
            )
            #div_loss = lambda_div * dwut.interp_loss(pred, y, mask_HR[0])
            ssim_loss = lambda_ssim * anti_clone_ssim(pred, y, mask_HR[0])

            loss = mse_loss + ssim_loss #+ div_loss 

            if WGHM:
                wghm = wghm.to(device)
                wghm_loss = lambda_wghm * masked_wghm_mse(pred, wghm, mask_HR[0])
                loss = loss + wghm_loss
                total_wghm += float(wghm_loss.item())
            else:
                wghm_loss = None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_mse += float(mse_loss.item())
            #total_div += float(div_loss.item())
            total_ssim += float(ssim_loss.item())
            total_loss += float(loss.item())

        n = max(len(dataloader), 1)
        if verbose or epoch == epochs - 1:
            if WGHM:
                logging.info(
                    "Epoch %d/%d | MSE: %.6f | SSIM: %.6f | WGAP: %.6f | loss_tot: %.6f",
                    epoch + 1,
                    epochs,
                    total_mse / n,
                    total_ssim / n,
                    total_wghm / n,
                    total_loss / n,
                )
            else:
                logging.info(
                    "Epoch %d/%d | MSE: %.6f | SSIM: %.6f | WGAP: N/A | loss_tot: %.6f",
                    epoch + 1,
                    epochs,
                    total_mse / n,
                    total_ssim / n,
                    total_loss / n,
                )
