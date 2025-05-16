import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from src.denoising_toy_utils import (
    load_model,
    p_sample_loop,
    create_diff_dict,
    extract,
    array_to_gif,
    device,
)

# Ensure light (default) matplotlib style
plt.style.use('default')

def parse_args():
    parser = argparse.ArgumentParser(description='Physics-Guided Sampling for Toy Hypersphere Example')
    parser.add_argument('--physics_model_path', type=str, default='./trained_models/toy/run_1',
                        help='Path to the physics-informed model directory (e.g. ./trained_models/toy/run_1)')
    parser.add_argument('--recon_model_path', type=str, default='./trained_models/toy/recon_only',
                        help='Path to the reconstruction-only model directory (e.g. ./trained_models/toy/recon_only)')
    parser.add_argument('--physics_model_step', type=int, default=400,
                        help='Checkpoint step of physics model to load (e.g. 400)')
    parser.add_argument('--recon_model_step', type=int, default=400,
                        help='Checkpoint step of reconstruction model to load (e.g. 400)')
    parser.add_argument('--omega', type=float, default=1.0, dest='guidance_scale',
                        help='Classifier-free guidance weight ω (0 = recon-only, 1 = physics model)')
    parser.add_argument('--guidance_schedule', type=str, default='constant', choices=['constant', 'linear'],
                        help='Schedule for guidance weight over timesteps. "constant" uses fixed ω, "linear" ramps ω from 0 to the specified value across diffusion steps.')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./toy_guided_samples',
                        help='Directory to store generated samples and figures')
    parser.add_argument('--create_gif', action='store_true', help='Create GIF of the guided trajectory')
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Guided sampling utilities (adapted from sample_guided.py for toy case)
# -----------------------------------------------------------------------------

def guided_p_sample(x, t, diff_dict, physics_model, recon_model, omega=1.0, surpress_noise=True):
    """Single reverse-diffusion step with classifier-free guidance.

    Args:
        x: Current noisy sample (batch, dim)
        t: Integer timestep (scalar)
        diff_dict: Dictionary with diffusion coefficients (from create_diff_dict)
        physics_model: Trained physics-informed network
        recon_model: Trained reconstruction-only network
        omega: Float
        surpress_noise: If True, no noise is added at final step (t==0)
    Returns:
        sample at timestep t-1
    """
    t_tensor = torch.tensor([t], device=x.device, dtype=torch.long)
    batch = x.shape[0]
    t_repeat = t_tensor.repeat(batch)

    # Both models predict x0 (model_pred_mode='x0')
    physics_pred = physics_model(x, t_repeat)
    recon_pred = recon_model(x, t_repeat)

    guided_pred = recon_pred + omega * (physics_pred - recon_pred)

    # Compute posterior mean as in p_sample (model_pred_mode == 'x0')
    mean = (
        extract(diff_dict['posterior_mean_coef1'], t_tensor, x) * guided_pred +
        extract(diff_dict['posterior_mean_coef2'], t_tensor, x) * x
    )

    # Sample
    z = torch.randn_like(x, device=x.device)
    sigma_t = extract(diff_dict['betas'], t_tensor, x).sqrt()
    nonzero_mask = (1. - (t_tensor == 0).float()) if surpress_noise else 1.
    sample = mean + nonzero_mask * sigma_t * z

    return sample, guided_pred.detach()


def guided_p_sample_loop(shape, n_steps, diff_dict, physics_model, recon_model, *,
                         omega=1.0, guidance_schedule='constant', surpress_noise=True):
    """Run the full guided reverse process returning the trajectory list using weight ω.

    Args:
        shape: Output tensor shape (batch, dim)
        n_steps: Total number of diffusion steps used by the model
        diff_dict: Diffusion dictionary
        physics_model, recon_model: Trained models
        omega: Target guidance strength (maximum value for scheduled variants)
        guidance_schedule: 'constant' or 'linear'. If 'linear', guidance ramps linearly
                           from 0 at the start of sampling (high-noise regime) to the
                           specified omega at the final step (low-noise regime).
        surpress_noise: Whether to suppress noise at t==0
    Returns:
        List of intermediate samples (including initial noise and final sample)
    """
    cur_x = torch.randn(shape, device=diff_dict['alphas'].device)
    x_seq = [cur_x.detach().cpu()]

    for i in reversed(range(n_steps)):
        # Determine the guidance weight for this timestep based on schedule
        if guidance_schedule == 'constant':
            omega_t = omega
        elif guidance_schedule == 'linear':
            # Linear ramp: 0 at the first reverse step (i == n_steps-1) -> omega at final step (i == 0)
            progress = 1.0 - (i / max(n_steps - 1, 1))
            omega_t = omega * progress
        else:
            raise ValueError(f"Unknown guidance_schedule '{guidance_schedule}'")

        cur_x, _ = guided_p_sample(cur_x.detach(), i, diff_dict, physics_model, recon_model,
                                    omega=omega_t, surpress_noise=surpress_noise)
        x_seq.append(cur_x.detach().cpu())

    return x_seq


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load models and associated diffusion dictionaries
    # ------------------------------------------------------------------
    # Check all possible locations for model checkpoints
    possible_physics_dirs = [
        Path(args.physics_model_path) / 'model',  # Standard location
        Path(args.physics_model_path)             # Direct checkpoint dir
    ]
    
    possible_recon_dirs = [
        Path(args.recon_model_path) / 'model',  # Standard location
        Path(args.recon_model_path)             # Direct checkpoint dir
    ]
    
    physics_ckpt = None
    for dir_path in possible_physics_dirs:
        candidate = dir_path / f'checkpoint_{args.physics_model_step}.pt'
        if candidate.exists():
            physics_ckpt = candidate
            break
    
    recon_ckpt = None
    for dir_path in possible_recon_dirs:
        candidate = dir_path / f'checkpoint_{args.recon_model_step}.pt'
        if candidate.exists():
            recon_ckpt = candidate
            break

    if physics_ckpt is None:
        raise FileNotFoundError(f'Physics model checkpoint not found for step {args.physics_model_step}')
    if recon_ckpt is None:
        raise FileNotFoundError(f'Reconstruction-only model checkpoint not found for step {args.recon_model_step}')
        
    print(f"Loading physics model from: {physics_ckpt}")
    print(f"Loading recon-only model from: {recon_ckpt}")

    physics_model, diff_dict_phys, n_steps_phys, dim_phys, pred_mode_phys, *_ = load_model(str(physics_ckpt))
    recon_model, diff_dict_recon, n_steps_recon, dim_recon, pred_mode_recon, *_ = load_model(str(recon_ckpt))

    # Move to device & eval
    physics_model = physics_model.to(device).eval()
    recon_model = recon_model.to(device).eval()

    # Verify models are on the correct device
    for m in (physics_model, recon_model):
        for param in m.parameters():
            assert param.device == device, "Model parameters not on target device"
        for buff in m.buffers():
            if torch.is_tensor(buff):
                assert buff.device == device, "Model buffers not on target device"

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if n_steps_phys != n_steps_recon:
        print('Warning: Mismatch in n_steps between models. Using physics model value.')
    if dim_phys != dim_recon:
        print('Warning: Mismatch in input dimension between models.')
    if pred_mode_phys != 'x0' or pred_mode_recon != 'x0':
        print('Warning: Expected model_pred_mode "x0" for both models.')

    n_steps = n_steps_phys
    dim = dim_phys
    # Move diffusion dictionary tensors to the model device (GPU/CPU)
    diff_dict = {}
    for k, v in diff_dict_phys.items():
        if torch.is_tensor(v):
            diff_dict[k] = v.to(device)
        else:
            diff_dict[k] = v

    # ------------------------------------------------------------------
    # Run sampling for each model and the guided method
    # ------------------------------------------------------------------
    sample_shape = (args.n_samples, dim)
    print(f'Generating {args.n_samples} samples for each method (dim={dim}, steps={n_steps})')

    with torch.no_grad():
        seq_recon = p_sample_loop(recon_model, sample_shape, n_steps, diff_dict,
                                  model_pred_mode='x0', save_output=False, surpress_noise=True)[0]  # take x_seq
        seq_phys = p_sample_loop(physics_model, sample_shape, n_steps, diff_dict,
                                 model_pred_mode='x0', save_output=False, surpress_noise=True)[0]
        seq_guided = guided_p_sample_loop(sample_shape, n_steps, diff_dict, physics_model, recon_model,
                                          omega=args.guidance_scale, guidance_schedule=args.guidance_schedule,
                                          surpress_noise=True)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    final_recon = seq_recon[-1].numpy()
    final_phys = seq_phys[-1].numpy()
    final_guided = seq_guided[-1].numpy()

    # ------------------------------------------------------------------
    # Quantitative evaluation: residual MAE and MSE (|‖x‖^2-1|)
    # ------------------------------------------------------------------
    def residual_stats(samples):
        res = np.sum(samples**2, axis=1) - 1.0  # deviation from unit norm
        abs_res = np.abs(res)
        return abs_res.mean(), (res**2).mean()

    mae_recon, mse_recon = residual_stats(final_recon)
    mae_phys,  mse_phys  = residual_stats(final_phys)
    mae_guid, mse_guid   = residual_stats(final_guided)

    print("\nResidual metrics (MAE | squared-error MSE):")
    print(f"Recon-only  : MAE={mae_recon:.4e}  MSE={mse_recon:.4e}")
    print(f"Physics-model: MAE={mae_phys :.4e}  MSE={mse_phys :.4e}")
    print(f"Guided       : MAE={mae_guid:.4e}  MSE={mse_guid:.4e}\n")

    # Create comparative visualization
    plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='grey', lw=1, label='Unit circle')
    plt.scatter(final_recon[:, 0], final_recon[:, 1], s=10, color='blue', label=f'Vanilla Diffusion Model (Physics Residual {mse_recon:.2e})')
    plt.scatter(final_phys[:, 0], final_phys[:, 1], s=10, color='purple', label=f'Physics-Informed Diffusion Model (Physics Residual  {mse_phys:.2e})')
    plt.scatter(final_guided[:, 0], final_guided[:, 1], s=10, color='green', label=f'Ours: Tuned-Physics Guidance Diffusion Model (Physics Residual  {mse_guid:.2e})')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower left', framealpha=0.85, fontsize=8)
    ax.set_title(f'Guided Sampling (ω={args.guidance_scale})')

    # Save as PNG; GIF creation handled separately
    plt.savefig('circular_samples.png')
    plt.close()

    # Optionally create GIF for guided trajectory
    if args.create_gif:
        seq_array = torch.stack(seq_guided, dim=0).numpy()
        # Determine axis limits from physics model final sample for consistency
        x_lim = (seq_array[:, :, 0].min(), seq_array[:, :, 0].max())
        y_lim = (seq_array[:, :, 1].min(), seq_array[:, :, 1].max())
        array_to_gif(seq_array, str(Path(args.output_dir) / 'guided_trajectory.gif'), x_lim=x_lim, y_lim=y_lim, label='guided')

    # Save final samples as csv
    np.savetxt(Path(args.output_dir) / 'samples_recon.csv', final_recon, delimiter=',')
    np.savetxt(Path(args.output_dir) / 'samples_physics.csv', final_phys, delimiter=',')
    np.savetxt(Path(args.output_dir) / 'samples_guided.csv', final_guided, delimiter=',')

    print(f'Sampling complete. Outputs saved to {args.output_dir}')


if __name__ == '__main__':
    main() 