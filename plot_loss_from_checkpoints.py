import os
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from src.data_utils import Dataset, cycle
from src.denoising_utils import DenoisingDiffusion
from src.residuals_darcy import ResidualsDarcy
from src.unet_model import Unet3D
from src.denoising_utils import save_model, load_model  # type: ignore

# -----------------------------------------------------------------------------
# Reset any custom matplotlib styling that might have been set by dependencies
# (e.g. src.denoising_utils uses a dark HDR style).  Users asked for "normal"
# Matplotlib defaults instead.
# -----------------------------------------------------------------------------
import matplotlib as mpl

# Restore Matplotlib default parameters
mpl.rcParams.update(mpl.rcParamsDefault)
# Optionally, switch to a clean style sheet explicitly
# plt.style.use("default")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints and plot loss curves (training-style) without needing the original log file."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./trained_models/recon_only",
        help="Directory that contains the 'model' sub-folder with checkpoint_XXXX.pt files and model.yaml",
    )
    parser.add_argument(
        "--dataset_split",
        choices=["train", "valid"],
        default="valid",
        help="Which dataset split to use for the loss evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for loss evaluation. Larger = faster but uses more memory.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum number of batches to evaluate per checkpoint (0 = all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'cpu', 'cuda', or 'auto' (use cuda if available).",
    )
    return parser.parse_args()


def get_dataset_paths(split: str) -> Tuple[str, str]:
    """Return csv paths for Darcy p and K fields."""
    base = Path("./data/darcy")
    p_path = base / split / "p_data.csv"
    k_path = base / split / "K_data.csv"
    if not p_path.exists() or not k_path.exists():
        raise FileNotFoundError(f"Could not find dataset files at {p_path} / {k_path}.")
    return str(p_path), str(k_path)


def collect_checkpoints(model_folder: Path) -> List[Tuple[Path, int]]:
    """Return list of (path, iteration) sorted by iteration."""
    checkpoints = []
    for ckpt in model_folder.glob("checkpoint_*.pt"):
        try:
            iter_str = ckpt.stem.split("_")[1]
            iteration = int(iter_str)
            checkpoints.append((ckpt, iteration))
        except (IndexError, ValueError):
            continue
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def evaluate_checkpoint(
    ckpt_path: Path,
    config: dict,
    dl: DataLoader,
    device: torch.device,
) -> float:
    """Load checkpoint, compute mean loss over a dataloader, return scalar."""
    # Create diffusion utilities and model fresh for each checkpoint to avoid state leakage
    diff_steps = config["diff_steps"]
    fd_acc = config["fd_acc"]
    c_data = config["c_data"]
    c_residual = config["c_residual"]
    c_ineq = config["c_ineq"]
    lambda_opt = config["lambda_opt"]

    diffusion_utils = DenoisingDiffusion(diff_steps, device)

    output_dim = 2  # Darcy setup (pressure & permeability)
    model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=False).to(device)
    load_model(ckpt_path, model)
    model.eval()

    # Residuals object (physics not needed for recon-only but function call requires it)
    residuals = ResidualsDarcy(
        model=model,
        fd_acc=fd_acc,
        pixels_per_dim=64,
        pixels_at_boundary=True,
        reverse_d1=True,
        device=device,
        bcs="none",
        domain_length=1.0,
    )

    losses = []
    batches_processed = 0
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            loss, *_ = diffusion_utils.model_estimation_loss(
                batch,
                residual_func=residuals,
                c_data=c_data,
                c_residual=c_residual,
                c_ineq=c_ineq,
                lambda_opt=lambda_opt,
            )
            losses.append(loss.item())
            batches_processed += 1
            if 0 < args.max_batches <= batches_processed:
                break
    return float(np.mean(losses)) if losses else np.nan


if __name__ == "__main__":
    args = parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_dir = Path(args.model_dir)
    model_folder = model_dir / "model"
    if not model_folder.exists():
        raise FileNotFoundError(f"Could not find model folder at {model_folder}")

    # Load config
    config_path = model_folder / "model.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"model.yaml not found in {model_folder}")
    import yaml  # local import to avoid dependency if not needed earlier

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Prepare dataset
    p_path, k_path = get_dataset_paths("valid" if args.dataset_split == "valid" else "train")
    ds = Dataset((p_path, k_path))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Gather checkpoints
    checkpoints = collect_checkpoints(model_folder)
    if not checkpoints:
        raise RuntimeError("No checkpoints found in the model directory.")

    iterations = []
    losses = []

    for ckpt_path, iteration in tqdm(checkpoints, desc="Evaluating checkpoints"):
        loss_val = evaluate_checkpoint(ckpt_path, config, dl, device)
        iterations.append(iteration)
        losses.append(loss_val)

    # Save results as CSV
    csv_path = model_dir / "loss_curve.csv"
    import pandas as pd

    pd.DataFrame({"iteration": iterations, "loss": losses}).to_csv(csv_path, index=False)

    # Plot curve
    plt.figure(figsize=(7, 5))
    plt.semilogy(iterations, losses, marker="o", linewidth=2)
    plt.xlabel("Iteration (checkpoint)")
    plt.ylabel("Loss (log scale)")
    plt.title("Validation Loss vs Training Iteration")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    fig_path = model_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Saved CSV to {csv_path}\nSaved figure to {fig_path}") 