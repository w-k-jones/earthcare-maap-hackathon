from __future__ import annotations

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


class WeightedMSELoss(torch.nn.Module):
    """
    Point-wise weighted MSE for sparse targets.

    Points with target > positive_threshold get weight 1 + positive_weight.
    With log1p targets, positive_threshold=0 still identifies original counts > 0.
    """

    def __init__(
        self,
        positive_weight: float = 10.0,
        positive_threshold: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")
        self.positive_weight = positive_weight
        self.positive_threshold = positive_threshold
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        weights = 1.0 + self.positive_weight * (target > self.positive_threshold).to(loss.dtype)
        loss = weights * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _prediction_to_target_shape(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Adapts the output of the model to the shape of y.

    Dataset:
        x: [B, C_in, height, along_track]
        y: [B, C_out, along_track]

    UNet:
        pred: [B, C_out, height, along_track]

    If the model still outputs a 2D map we collapse the height dimension.
    """
    if pred.shape == y.shape:
        return pred

    if pred.ndim == 4 and y.ndim == 3:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, y.shape[-1]))
        return pred.squeeze(2)

    raise ValueError(f"Output shape {tuple(pred.shape)} incompatible with target {tuple(y.shape)}")


def _plot_fixed_prediction_target(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    sample_idx: int,
    target_channel: int,
    epoch_idx: int,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    dataset = dataloader.dataset
    if len(dataset) == 0:
        return

    sample_idx = min(sample_idx, len(dataset) - 1)
    sample = dataset[sample_idx]

    x = sample["x"].unsqueeze(0).to(device)
    y = sample["y"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = _prediction_to_target_shape(pred, y)

    pred_1d = pred[0, target_channel].detach().cpu().numpy()
    y_1d = y[0, target_channel].detach().cpu().numpy()
    if getattr(dataset, "target_log1p", False):
        pred_1d = np.expm1(pred_1d)
        y_1d = np.expm1(y_1d)
        pred_1d = np.clip(pred_1d, a_min=0.0, a_max=None)
        y_1d = np.clip(y_1d, a_min=0.0, a_max=None)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_1d, label="target", linewidth=2.0)
    ax.plot(pred_1d, label="prediction", linewidth=1.5)
    ax.set_title(f"Prediction vs target | epoch {epoch_idx}")
    ax.set_xlabel("along_track")
    ylabel = f"target channel {target_channel}"
    if getattr(dataset, "target_log1p", False):
        ylabel += " (count scale)"
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = sample.get("path")
    if path is not None:
        fig.suptitle(os.path.basename(path), fontsize=10)

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"prediction_target_epoch_{epoch_idx:03d}.png"),
            dpi=150,
        )

    if show:
        plt.show()

    plt.close(fig)


def _run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    device: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        if n_batches == 0 and is_train:
            print(
                f"First train batch devices | x: {x.device} | y: {y.device} | "
                f"model: {next(model.parameters()).device}",
                flush=True,
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            pred = _prediction_to_target_shape(pred, y)
            loss = criterion(pred, y)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model: torch.nn.Module,
    datamodule=None,
    train_dataloader: DataLoader | None = None,
    val_dataloader: DataLoader | None = None,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str | None = None,
    criterion: torch.nn.Module | None = None,
    plot: bool = True,
    plot_every: int = 1,
    plot_sample_idx: int = 0,
    plot_target_channel: int = 0,
    plot_save_dir: str | None = None,
    show_plot: bool = True,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Training loop compatible with EarthCARELightningDataModule/EarthCARELightningDataset.

    Uses batch dict:
        batch["x"] -> [B, C_in, height, along_track]
        batch["y"] -> [B, C_out, along_track]
    """
    if datamodule is not None:
        if verbose:
            print("Loading dataset and computing normalization stats...", flush=True)
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        if verbose:
            train_size = len(datamodule.train_dataset) if datamodule.train_dataset is not None else 0
            val_size = len(datamodule.val_dataset) if datamodule.val_dataset is not None else 0
            test_size = len(datamodule.test_dataset) if datamodule.test_dataset is not None else 0
            print(
                "Dataset loaded | "
                f"train: {train_size} | val: {val_size} | test: {test_size}",
                flush=True,
            )

    if train_dataloader is None:
        raise ValueError("Pass datamodule or train_dataloader.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if verbose:
        print(f"Training device: {device}", flush=True)
        print(f"Model parameter device: {next(model.parameters()).device}", flush=True)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = criterion or torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    plot_dataloader = train_dataloader

    for epoch in range(epochs):
        epoch_start = time.time()

        if plot and plot_every > 0 and epoch % plot_every == 0:
            _plot_fixed_prediction_target(
                model=model,
                dataloader=plot_dataloader,
                device=device,
                sample_idx=plot_sample_idx,
                target_channel=plot_target_channel,
                epoch_idx=epoch,
                save_dir=plot_save_dir,
                show=show_plot,
            )

        train_loss = _run_epoch(model, train_dataloader, optimizer, criterion, device)
        history["train_loss"].append(train_loss)

        if val_dataloader is not None:
            with torch.no_grad():
                val_loss = _run_epoch(model, val_dataloader, None, criterion, device)
            history["val_loss"].append(val_loss)
        else:
            val_loss = None

        if verbose:
            elapsed = time.time() - epoch_start
            if val_loss is None:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"train_loss: {train_loss:.6f} | "
                    f"time: {elapsed:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"train_loss: {train_loss:.6f} | "
                    f"val_loss: {val_loss:.6f} | "
                    f"time: {elapsed:.1f}s",
                    flush=True,
                )

    if plot:
        _plot_fixed_prediction_target(
            model=model,
            dataloader=plot_dataloader,
            device=device,
            sample_idx=plot_sample_idx,
            target_channel=plot_target_channel,
            epoch_idx=epochs,
            save_dir=plot_save_dir,
            show=show_plot,
        )

    return history

### Example ###
# from datamodule import EarthCARELightningDataModule
# from models.unet import UNet
# from train import train

# dm = EarthCARELightningDataModule(
#     data_dir="/path/to/patches",
#     input_vars=["var1", "var2"],
#     target_vars=["lightning_target"],
#     batch_size=8,
# )

# model = UNet(in_channels=len(dm.input_vars), out_channels=len(dm.target_vars))

# history = train(
#     model=model,
#     datamodule=dm,
#     epochs=20,
#     lr=1e-4,
# )
