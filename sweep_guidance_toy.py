import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm

from src.denoising_toy_utils import load_model, p_sample_loop, device
from sample_toy_guided import guided_p_sample_loop


def compute_mae_mse(samples: np.ndarray):
    res = np.sum(samples ** 2, axis=1) - 1.0
    mae = np.abs(res).mean()
    mse = (res ** 2).mean()
    return mae, mse


def main():
    physics_model_path = './trained_models/toy/run_1'
    recon_model_path = './trained_models/toy/recon_only'
    physics_step = 400
    recon_step = 400

    # Load models once
    physics_ckpt = Path(physics_model_path) / 'model' / f'checkpoint_{physics_step}.pt'
    recon_ckpt = Path(recon_model_path) / 'model' / f'checkpoint_{recon_step}.pt'

    physics_model, diff_dict_phys, n_steps, dim, *_ = load_model(str(physics_ckpt))
    recon_model, _, _, _, *_ = load_model(str(recon_ckpt))

    physics_model = physics_model.to(device).eval()
    recon_model = recon_model.to(device).eval()

    # move diff_dict to correct device
    diff_dict = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in diff_dict_phys.items()}

    sample_shape = (1000, dim)

    # Baselines
    with torch.no_grad():
        seq_recon = p_sample_loop(recon_model, sample_shape, n_steps, diff_dict, model_pred_mode='x0', save_output=False, surpress_noise=True)[0]
        seq_phys = p_sample_loop(physics_model, sample_shape, n_steps, diff_dict, model_pred_mode='x0', save_output=False, surpress_noise=True)[0]
    recon_mae, recon_mse = compute_mae_mse(seq_recon[-1].cpu().numpy())
    phys_mae, phys_mse = compute_mae_mse(seq_phys[-1].cpu().numpy())

    # Sweep guidance strengths
    omegas = np.arange(0.0, 2.01, 0.1)
    maes = []
    mses = []

    print("Sweeping guidance scale...")
    for omega in tqdm(omegas):
        with torch.no_grad():
            seq_guided = guided_p_sample_loop(sample_shape, n_steps, diff_dict, physics_model, recon_model, guidance_scale=float(omega), surpress_noise=True)
        mae, mse = compute_mae_mse(seq_guided[-1].cpu().numpy())
        maes.append(mae)
        mses.append(mse)
        print(f"ω={omega:.1f}: MAE={mae:.4e}  MSE={mse:.4e}")

    # Use default matplotlib style for a simple clean look
    plt.style.use('default')

    # ------------------- MAE plot -------------------
    plt.figure(figsize=(7, 4))
    handles = []
    # Plot in drawing order (but legend will be reordered)
    h3, = plt.plot(omegas, maes, '-o', color='forestgreen', label='Guided MAE')
    min_idx = int(np.argmin(maes))
    h4 = plt.scatter(omegas[min_idx], maes[min_idx], s=80, c='red', zorder=5,
                label=f'Min guided MAE = {maes[min_idx]:.2e} @ ω={omegas[min_idx]:.1f}')
    h1 = plt.axhline(recon_mae, linestyle='--', color='dodgerblue', label=f'Recon MAE = {recon_mae:.2e}')
    h2 = plt.axhline(phys_mae, linestyle='--', color='mediumorchid', label=f'Physics MAE = {phys_mae:.2e}')

    plt.xlabel('Guidance scale ω')
    plt.ylabel('MAE (|‖x‖²−1|)')
    plt.title('Guided MAE vs guidance scale')
    plt.legend(handles=[h1, h2, h3, h4], fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('guidance_mae_sweep.png', dpi=300)

    # ------------------- MSE plot -------------------
    plt.figure(figsize=(7, 4))
    handles = []
    # Plot in drawing order (but legend will be reordered)
    h3, = plt.plot(omegas, mses, '-o', color='forestgreen', label='Guided MSE')
    min_idx = int(np.argmin(mses))
    h4 = plt.scatter(omegas[min_idx], mses[min_idx], s=80, c='red', zorder=5,
                label=f'Min guided MSE = {mses[min_idx]:.2e} @ ω={omegas[min_idx]:.1f}')
    h1 = plt.axhline(recon_mse, linestyle='--', color='dodgerblue', label=f'Recon MSE = {recon_mse:.2e}')
    h2 = plt.axhline(phys_mse, linestyle='--', color='mediumorchid', label=f'Physics MSE = {phys_mse:.2e}')

    plt.xlabel('Guidance scale ω')
    plt.ylabel('MSE ((‖x‖²−1)²)')
    plt.title('Guided MSE vs guidance scale')
    plt.legend(handles=[h1, h2, h3, h4], fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('guidance_mse_sweep.png', dpi=300)

    print("Saved guidance_mae_sweep.png and guidance_mse_sweep.png")

if __name__ == '__main__':
    main() 