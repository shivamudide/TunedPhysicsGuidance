import os, yaml, argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy

def parse_args():
    parser = argparse.ArgumentParser(description='Physics-Guided Sampling for Diffusion Models')
    parser.add_argument('--physics_model_path', type=str, default='./trained_models/darcy/PIDM-ME',
                        help='Path to the physics-informed model')
    parser.add_argument('--recon_model_path', type=str, default='./trained_models/recon_only',
                        help='Path to the reconstruction-only model')
    parser.add_argument('--physics_model_step', type=int, default=300000,
                        help='Training step for physics model checkpoint')
    parser.add_argument('--recon_model_step', type=int, default=10000,
                        help='Training step for reconstruction model checkpoint')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Scale for classifier-free guidance (1.0 = no guidance)')
    parser.add_argument('--dynamic_guidance', action='store_true',
                        help='Scale guidance per sample based on residual mismatch')
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./guided_samples',
                        help='Output directory for samples')
    parser.add_argument('--create_gif', action='store_true',
                        help='Create GIFs of the denoising process')
    parser.add_argument('--schedule', choices=['none','linear','cosine'], default='none',
                        help='timestep schedule for guidance strength')
    parser.add_argument('--post_correction_iters', type=int, default=0,
                        help='extra residual correction iterations after sampling')
    parser.add_argument('--extra_steps', type=int, default=0,
                        help='number of additional refinement steps')
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='refinement step size')
    parser.add_argument('--smooth_weight', type=float, default=0.0,
                        help='weight for smoothness (total variation) regularization during extra refinement')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create descriptive output directory
    output_dir = f"{args.output_dir}/physics_{args.physics_model_step}_recon_{args.recon_model_step}_guidance_{args.guidance_scale:.1f}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using physics model: {args.physics_model_path} (checkpoint {args.physics_model_step})")
    print(f"Using reconstruction model: {args.recon_model_path} (checkpoint {args.recon_model_step})")
    print(f"Guidance scale: {args.guidance_scale}")
    
    # Load physics model config
    physics_config_path = Path(args.physics_model_path, 'model', 'model.yaml')
    if not physics_config_path.exists():
        print(f"Error: Physics model config not found at {physics_config_path}")
        return
    physics_config = yaml.safe_load(physics_config_path.read_text())
    
    # Load reconstruction model config
    recon_config_path = Path(args.recon_model_path, 'model', 'model.yaml')
    if not recon_config_path.exists():
        print(f"Error: Reconstruction model config not found at {recon_config_path}")
        return
    recon_config = yaml.safe_load(recon_config_path.read_text())
    
    # Parameters from configs
    gov_eqs = physics_config['gov_eqs']
    fd_acc = physics_config['fd_acc']
    diff_steps = physics_config['diff_steps']
    residual_grad_guidance = physics_config['residual_grad_guidance']
    correction_mode = physics_config['correction_mode']
    M_correction = physics_config['M_correction']
    N_correction = physics_config['N_correction']
    
    # Validate we're using Darcy
    if gov_eqs != 'darcy':
        print(f"Error: This script currently only supports Darcy flow models, got {gov_eqs}")
        return
    
    # Setup problem parameters
    use_ddim_x0 = False
    ddim_steps = 0
    input_dim = 2
    output_dim = 2
    pixels_per_dim = 64  # IMPORTANT: This is 64x64 resolution
    pixels_at_boundary = True
    domain_length = 1.
    reverse_d1 = True
    bcs = 'none'
    return_optimizer = False
    return_inequality = False
    sigmoid_last_channel = False
    
    # Initialize diffusion utils
    diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)
    
    # Initialize both models
    physics_model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)
    recon_model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)
    
    # Load model checkpoints
    physics_checkpoint_path = Path(args.physics_model_path, 'model', f'checkpoint_{args.physics_model_step}.pt')
    recon_checkpoint_path = Path(args.recon_model_path, 'model', f'checkpoint_{args.recon_model_step}.pt')
    
    if not physics_checkpoint_path.exists():
        print(f"Error: Physics model checkpoint not found at {physics_checkpoint_path}")
        return
    if not recon_checkpoint_path.exists():
        print(f"Error: Reconstruction model checkpoint not found at {recon_checkpoint_path}")
        return
    
    load_model(physics_checkpoint_path, physics_model)
    load_model(recon_checkpoint_path, recon_model)
    
    # Initialize residuals object with physics model
    residuals = ResidualsDarcy(
        model=physics_model, fd_acc=fd_acc, pixels_per_dim=pixels_per_dim,
        pixels_at_boundary=pixels_at_boundary, reverse_d1=reverse_d1,
        device=device, bcs=bcs, domain_length=domain_length,
        residual_grad_guidance=residual_grad_guidance,
        use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps
    )
    
    def correct_again(x, iterations=100, lr=0.01, smooth_weight=0.0):
        
        # Convert to model format and create a copy with gradients
        x_optim = generalized_image_to_b_xy_c(x).clone().detach().to(device)
        x_optim.requires_grad_(True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([x_optim], lr=lr)
        
        # Create time tensor (t=0 for fully denoised)
        batch_size = x.shape[0]
        t_tensor = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        # Helper: total variation loss for smoothness
        def tv_loss(img):
            # img shape: (batch, channels, H, W)
            diff_x = img[:, :, 1:, :] - img[:, :, :-1, :]
            diff_y = img[:, :, :, 1:] - img[:, :, :, :-1]
            return (diff_x.abs().mean() + diff_y.abs().mean())
        
        # Initial residual computation (no print)
        with torch.no_grad():
            # Format input as expected by compute_residual: ((x, t), ...)
            model_input = (x_optim, t_tensor)
            out_dict = residuals.compute_residual(
                (model_input, ), reduce='per-batch', return_model_out=True)
        
        # Optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Compute residual
            model_input = (x_optim, t_tensor)
            out_dict = residuals.compute_residual(
                (model_input, ), reduce='per-batch', return_model_out=True)
            residual = out_dict['residual'].abs().mean()
            
            loss = residual
            if smooth_weight > 0.0:
                img_rep = generalized_b_xy_c_to_image(x_optim)
                loss = loss + smooth_weight * tv_loss(img_rep)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
        
        # Final evaluation
        with torch.no_grad():
            model_input = (x_optim, t_tensor)
            out_dict = residuals.compute_residual(
                (model_input, ), reduce='per-batch', return_model_out=True)
        
        # Convert back to image format
        x_optimized = generalized_b_xy_c_to_image(x_optim.detach())
        return x_optimized, out_dict['residual']
    
    # Count parameters
    num_params = sum(p.numel() for p in physics_model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')
    
    # Create custom p_sample function with guidance
    def guided_p_sample(x, conditioning_input, t, guidance_scale=1.0, save_output=False, 
                        surpress_noise=False, use_dynamic_threshold=False, eval_residuals=False,
                        return_optimizer=False, return_inequality=False, residual_correction=False,
                        dynamic_guidance=False):
        
        x_init = x.clone().detach()
        if conditioning_input is not None:
            conditioning, bcs, solution = conditioning_input
            x_cond = torch.cat((x, conditioning), dim=1)
        else:
            x_cond = x
        
        batch_size = len(x)
        t_tensor = torch.tensor([t], device=x.device)
        
        # Convert to format expected by models
        model_input = image_to_b_xy_c(x_cond)
        model_input = (model_input, t_tensor.repeat(batch_size))
        
        # Store original model in residual_func
        original_model = residuals.model
        
        # Get prediction from physics model
        residuals.model = physics_model
        physics_input = (model_input, )
        physics_out_dict = residuals.compute_residual(
            physics_input, reduce='per-batch', return_model_out=True,
            return_optimizer=return_optimizer, return_inequality=return_inequality,
            sample=(t == 0), ddim_func=diffusion_utils.ddim_sample_x0
        )
        physics_output, physics_residual = physics_out_dict['model_out'], physics_out_dict['residual']
        
        # Get prediction from reconstruction-only model
        residuals.model = recon_model
        recon_input = (model_input, )
        recon_out_dict = residuals.compute_residual(
            recon_input, reduce='per-batch', return_model_out=True,
            return_optimizer=False, return_inequality=False, 
            sample=(t == 0), ddim_func=diffusion_utils.ddim_sample_x0
        )
        recon_output = recon_out_dict['model_out']
        recon_residual = recon_out_dict['residual'] if 'residual' in recon_out_dict else None
        
        # Apply classifier-free guidance: output = recon_output + guidance_scale * (physics_output - recon_output)
        # When guidance_scale = 0.0, use just the reconstruction model output
        # When guidance_scale = 1.0, use just the physics model output
        if dynamic_guidance and recon_residual is not None:
            # Compute per-sample residual magnitude (mean absolute)
            res_phys = physics_residual.abs().mean(dim=tuple(range(1, physics_residual.ndim)))  # (batch,)
            res_recon = recon_residual.abs().mean(dim=tuple(range(1, recon_residual.ndim)))   # (batch,)
            ratio = (res_recon + 1e-8) / (res_phys + 1e-8)
            # Higher ratio => physics residual much lower => stronger guidance up to guidance_scale
            scale = torch.clamp(guidance_scale * ratio, 0., guidance_scale * 5.0)  # cap
            # reshape for broadcasting (batch,1,1,1)
            while scale.ndim < physics_output.ndim:
                scale = scale.unsqueeze(-1)
            guided_output = recon_output + scale * (physics_output - recon_output)
        else:
            if guidance_scale == 0.0:
                guided_output = recon_output
            else:
                guided_output = recon_output + guidance_scale * (physics_output - recon_output)
        
        # Convert to image format if needed
        if len(guided_output.shape) == 3:
            guided_output = generalized_b_xy_c_to_image(guided_output)
        
        # Apply residual correction if needed (always use physics model for corrections)
        residuals.model = physics_model  # Use physics model for correction
        if residual_correction and correction_mode == 'x0':
            guided_output, physics_residual = residuals.residual_correction(
                generalized_image_to_b_xy_c(guided_output)
            )
            guided_output = generalized_b_xy_c_to_image(guided_output)
        
        # Save intermediate output if requested
        model_intermediate = guided_output.clone().detach() if save_output else None
        
        # Compute mean for next sample
        mean = (
            extract(diffusion_utils.diff_dict['posterior_mean_coef1'], t_tensor, x_init) * guided_output +
            extract(diffusion_utils.diff_dict['posterior_mean_coef2'], t_tensor, x_init) * x_init
        )
        
        # Generate noise
        z = torch.randn_like(x_init, device=x.device)
        sigma_t = extract(diffusion_utils.diff_dict['betas'], t_tensor, x_init).sqrt()
        
        # Apply noise mask
        nonzero_mask = (1. - (t_tensor == 0).float()) if surpress_noise else 1.
        sample = mean + nonzero_mask * sigma_t * z
        
        # Apply residual correction to sample if needed (always use physics model for corrections)
        residuals.model = physics_model  # Use physics model for correction
        if residual_correction and correction_mode == 'xt':
            sample, physics_residual = residuals.residual_correction(
                generalized_image_to_b_xy_c(sample)
            )
            sample = generalized_b_xy_c_to_image(sample)
        
        # Restore original model
        residuals.model = original_model
        
        # Return evaluation metrics for the final step
        if (t == 0 and eval_residuals):
            aux_out = {'residual': physics_residual}
            if return_optimizer:
                aux_out['optimized_quant'] = physics_out_dict['optimizer']
            if return_inequality:
                aux_out['inequality_quant'] = physics_out_dict['inequality']
            return (sample, model_intermediate), aux_out
        else:
            return (sample, model_intermediate), None
    
    # Create guided sampling loop
    def timestep_scale(t, total, schedule):
        if schedule=='none':
            return 1.0
        if schedule=='linear':
            return 1.0 - t/total
            # return 0.0 + t/total
        if schedule=='cosine':
            import math
            # return -1*math.cos(0.5*math.pi*t/total) + 1.0
            return math.cos(0.5*math.pi*t/total) 
        return 1.0
    
    def guided_p_sample_loop(conditioning_input, shape, guidance_scale=1.0, save_output=False, 
                            surpress_noise=True, use_dynamic_threshold=False,
                            eval_residuals=False, return_optimizer=False, return_inequality=False):
        
        cur_x = torch.randn(shape, device=diffusion_utils.diff_dict['alphas'].device)
        x_seq = [cur_x.detach().cpu()]
        
        if save_output:
            interm_imgs = [torch.zeros(shape)]
        else:
            interm_imgs = []
        
        interm_img = None
        for i in reversed(range(diffusion_utils.n_steps)):
            
            residual_correction = False
            if i < N_correction:
                residual_correction = True
                eval_residuals = True
                
            step_fac = timestep_scale(i, diffusion_utils.n_steps-1, args.schedule)
            output = guided_p_sample(
                cur_x.detach(), conditioning_input, i, guidance_scale*step_fac,
                save_output, surpress_noise, use_dynamic_threshold,
                eval_residuals=eval_residuals, return_optimizer=return_optimizer,
                return_inequality=return_inequality, residual_correction=residual_correction,
                dynamic_guidance=args.dynamic_guidance
            )
            
            cur_x, interm_img = output[0]
            
            x_seq.append(cur_x.detach().cpu())
            if interm_img is not None:
                interm_imgs.append(interm_img.detach().cpu())
            else:
                interm_imgs.append(None)
        
        
        original_model = residuals.model
        residuals.model = physics_model  
        
        for i in range(M_correction):
            cur_x, residual = residuals.residual_correction(generalized_image_to_b_xy_c(cur_x))
            cur_x = generalized_b_xy_c_to_image(cur_x)
            x_seq.append(cur_x.detach().cpu())
            
            if eval_residuals and i == M_correction - 1:
                output[1]['residual'] = residual
                
        residuals.model = original_model
        
        
        if args.post_correction_iters > 0:
            residuals.model = physics_model
            for _ in range(args.post_correction_iters):
                cur_x, _ = residuals.residual_correction(generalized_image_to_b_xy_c(cur_x))
                cur_x = generalized_b_xy_c_to_image(cur_x)
            x_seq.append(cur_x.detach().cpu())
            residuals.model = physics_model
        
        if args.extra_steps > 0:
            residuals.model = physics_model
            cur_x, final_residual = correct_again(cur_x, iterations=args.extra_steps, lr=args.step_size, smooth_weight=args.smooth_weight)
            x_seq.append(cur_x.detach().cpu())
            # Update the final residual
            if eval_residuals:
                output[1]['residual'] = final_residual
        
        if eval_residuals:
            return (x_seq, interm_imgs), output[1]
        else:
            return x_seq, interm_imgs
    
    # Set up conditioning and shape based on problem
    conditioning_input = None
    # Make sure sample_shape has correct dimensions: [batch_size, channels, height, width]
    # For Darcy problem, this should be [n_samples, 2, 64, 64]
    sample_shape = (args.n_samples, output_dim, pixels_per_dim, pixels_per_dim)
    print(f"Generating samples with shape: {sample_shape}")
    
    # Run guided sampling
    print(f"Generating {args.n_samples} samples with guidance scale {args.guidance_scale}...")
    output = guided_p_sample_loop(
        conditioning_input, sample_shape, args.guidance_scale,
        save_output=True, surpress_noise=True, 
        use_dynamic_threshold=False, 
        eval_residuals=True, 
        return_optimizer=return_optimizer, 
        return_inequality=return_inequality
    )
    
    if output[1] is not None:
        seqs = output[0][0]
        # Store full residual tensor for visualization (shape: [batch, pixels^2, n_channels])
        residual_full = output[1]['residual']

        # Compute scalar statistics from residual for logging
        residual_stats = residual_full.abs().mean(dim=tuple(range(1, residual_full.ndim)))  # reduce to batch dim
    else:
        seqs = output[0]
        
    # Determine pre- and post-refinement predictions
    final_pred_tensor = seqs[-1]  # after extra refinement (or last state if none)
    if args.extra_steps > 0:
        pre_pred_tensor = seqs[-2]  # state immediately before extra refinement
    else:
        pre_pred_tensor = final_pred_tensor

    pre_pred = pre_pred_tensor.detach().cpu().numpy()
    final_pred = final_pred_tensor.detach().cpu().numpy()

    n_batch, n_channels, _, _ = final_pred.shape

    for b in range(n_batch):
        sample_dir = Path(output_dir) / f"sample_{b}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for c in range(n_channels):
            # -------------------- BEFORE refinement --------------------
            img_arr_pre = pre_pred[b, c]
            np.savetxt(sample_dir / f"sample_{c}_pre.csv", img_arr_pre, delimiter=',')
            norm_pre = (img_arr_pre - img_arr_pre.min()) / (img_arr_pre.max() - img_arr_pre.min() + 1e-8)
            plt.imsave(sample_dir / f"field_{c}_before.png", norm_pre, cmap='viridis', vmin=0, vmax=1)

            # -------------------- AFTER refinement ---------------------
            img_arr_post = final_pred[b, c]
            np.savetxt(sample_dir / f"sample_{c}_post.csv", img_arr_post, delimiter=',')
            norm_post = (img_arr_post - img_arr_post.min()) / (img_arr_post.max() - img_arr_post.min() + 1e-8)
            plt.imsave(sample_dir / f"field_{c}_after.png", norm_post, cmap='viridis', vmin=0, vmax=1)

            # optional GIF of the whole trajectory for this channel
            if args.create_gif:
                seq_arr = np.stack([frame[b, c].detach().cpu().numpy() for frame in seqs])  # (T, 64, 64)
                image_array_to_gif(seq_arr, sample_dir / f"field_{c}.gif")

        # ------------------------------------------------------------------
        # Save residual map (per-sample)
        # ------------------------------------------------------------------
        if 'residual_full' in locals():
            # Convert residual tensor to image format: shape (batch, n_res_channels, H, W)
            residual_img = generalized_b_xy_c_to_image(residual_full)[b]  # (n_res_channels, H, W)

            # Aggregate channels to single magnitude map for visualization
            residual_mag = residual_img.abs().mean(dim=0)  # (H, W)

            # Convert to numpy for saving
            residual_np = residual_mag.detach().cpu().numpy()

            # Save raw residual magnitude values
            np.savetxt(sample_dir / "residual.csv", residual_np, delimiter=',')

            # Normalise and save PNG (use perceptually uniform colormap)
            res_norm = (residual_np - residual_np.min()) / (residual_np.max() - residual_np.min() + 1e-8)
            plt.imsave(sample_dir / "residual.png", res_norm, cmap='viridis', vmin=0, vmax=1)

    if 'residual_stats' in locals():
        residuals_array = residual_stats.detach().cpu().numpy()
        
        # Use actual number of samples for statistics
        actual_samples = min(args.n_samples, len(residuals_array))
        sample_indices = list(range(actual_samples))
        
        # Create statistics dataframe
        df_data = {
            'Sample Index': sample_indices + ['Mean'],
            'Residuals (abs)': list(residuals_array[:actual_samples]) + [np.nanmean(residuals_array[:actual_samples])]
        }
        
        # Save statistics to CSV
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, 'sample_statistics.csv'), index=False)
        
        # Print summary
        print(f"Sampling completed. Results saved to {output_dir}")
        print(f"Average residual: {np.nanmean(residuals_array[:actual_samples]):.4e}")
        print(f"Min residual: {np.nanmin(residuals_array[:actual_samples]):.4e}")
        print(f"Max residual: {np.nanmax(residuals_array[:actual_samples]):.4e}")

if __name__ == "__main__":
    main() 