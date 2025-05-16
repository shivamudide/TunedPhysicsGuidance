import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from src.denoising_toy_utils import *

# Fix seeds for reproducibility
fix_seeds()

# Create both physics and recon-only models
configs = {
    'physics_model': {
        'name': 'physics_model',
        'x0_estimation': 'sample', 
        'reduced_ddim_steps': 0,
        'model_pred_mode': 'x0',
        'c_data': 1.0,
        'c_residual': 0.1,  # High residual weight for physics-informed
        'c_ineq': 0.0, 
        'lambda_opt': 0.0,
        'true_randomness': False,
        'dim': 2,
        'n_steps': 100,
        'use_dynamic_threshold': False,
        'train_num_steps': 400,
        'batch_size': 128,
        'no_samples': 1000,
        'sample_freq': 10,
        'tot_eval_steps': 11,
        'fix_axes': True,
        'save_output': True,
        'create_gif': False,
        'wandb_track': False  # Disable wandb
    },
    'recon_only': {
        'name': 'recon_only',
        'x0_estimation': 'sample',
        'reduced_ddim_steps': 0,
        'model_pred_mode': 'x0',
        'c_data': 1.0,
        'c_residual': 0.0,  # No residual weight for reconstruction-only
        'c_ineq': 0.0,
        'lambda_opt': 0.0,
        'true_randomness': False,
        'dim': 2,
        'n_steps': 100,
        'use_dynamic_threshold': False,
        'train_num_steps': 400,
        'batch_size': 128,
        'no_samples': 1000,
        'sample_freq': 10,
        'tot_eval_steps': 11,
        'fix_axes': True,
        'save_output': True,
        'create_gif': False,
        'wandb_track': False  # Disable wandb
    }
}

print("Starting training of toy models...")

# Make sure output directories exist
os.makedirs('./trained_models/toy', exist_ok=True)

# Generate data (same for both models)
data = sample_hypersphere(10**4, 2)  # 2D circle
dataset = torch.tensor(data).float().to(device)

# Print device info
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Functions for the residual, inequality and optimization
class ResidualFunc(nn.Module):
    '''Simple residual given by the unit hypersphere.'''
    def forward(self, x):
        return torch.sum(x**2, dim=1) - 1.0

class InequalityFunc(nn.Module):
    '''Simple example for experimentation.'''
    def __init__(self, threshold, mode='leq'):
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        density = torch.sum(torch.abs(x), dim=1)
        shift = density - self.threshold
        return self.relu(shift if self.mode == 'leq' else -shift), density

class OptimizationFunc(nn.Module):
    '''Simple example for experimentation.'''
    def forward(self, x):
        return x[:,0]

residual_func = ResidualFunc()
ineq_func = InequalityFunc(threshold=1.0, mode='leq')
opt_func = OptimizationFunc()

def log_metrics(metrics, step):
    # Simple no-op or print to console
    print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

def evaluate_and_log(seq, config):
    residual = residual_func(seq[0][-1]).abs().mean().item()
    metrics = {'residual_samples': residual}
    
    if config['c_ineq'] > 0:
        ineq_density = ineq_func(seq[0][-1])[1].abs().mean().item()
        metrics['density_samples'] = ineq_density
    
    if config['lambda_opt'] > 0:
        opt = opt_func(seq[0][-1]).abs().mean().item()
        metrics['opt_samples'] = opt
    
    return metrics

# Train each model
for model_name, config in configs.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name} model...")
    print(f"{'='*50}")
    
    # Derived config
    config['use_ddim_x0'] = config['x0_estimation'] == 'sample'
    output_save_dir = f'./trained_models/toy/{config["name"]}'
    os.makedirs(output_save_dir, exist_ok=True)
    
    # Define model and optimizer
    model = ConditionalModel(config['dim'], config['n_steps']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5.e-4)
    diff_dict = create_diff_dict(config['n_steps'], device)
    
    # Evaluation steps
    eval_steps = np.linspace(0, config['n_steps'], config['tot_eval_steps']).astype(int)
    
    # Training loop
    pbar = tqdm(range(config['train_num_steps'] + 1))
    for t in pbar:
        permutation = torch.randperm(dataset.size()[0])
        for i in range(0, dataset.size()[0], config['batch_size']):
            indices = permutation[i:i + config['batch_size']]
            batch_x = dataset[indices]
            if config['true_randomness']:
                batch_x = torch.randn_like(batch_x)
            
            loss, data_loss, residual_loss, ineq_loss, opt_loss = model_estimation_loss(
                model, batch_x, config['n_steps'], diff_dict, model_pred_mode=config['model_pred_mode'],
                residual_func=residual_func, ineq_func=ineq_func, opt_func=opt_func,
                c_data=config['c_data'], c_residual=config['c_residual'], c_ineq=config['c_ineq'],
                lambda_opt=config['lambda_opt'], use_ddim_x0=config['use_ddim_x0'], reduced_ddim_steps=config['reduced_ddim_steps'])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_description(f'training loss: {loss.item():.4f}')
        
        # Sample and visualize 
        if t % config['sample_freq'] == 0:
            shape = [config['no_samples'], config['dim']]
            seqs = p_sample_loop(
                model, shape, config['n_steps'], diff_dict,
                model_pred_mode=config['model_pred_mode'], save_output=config['save_output'],
                surpress_noise=True, use_dynamic_threshold=config['use_dynamic_threshold'],
                reduced_ddim_steps=config['reduced_ddim_steps'])
            
            metrics = evaluate_and_log(seqs, config)
            log_metrics(metrics, t)
            
            # Create visualization
            fig, axs = plt.subplots(1, config['tot_eval_steps'], figsize=(3 * config['tot_eval_steps'] - 3, 3))
            labels = ['sample', 'model_output', 'x0_estimate']
            for seq_idx, seq in enumerate(seqs):
                if seq:
                    for i_idx, i in enumerate(eval_steps):
                        cur_x = seq[i].detach().cpu()
                        if config['fix_axes'] and seq_idx == 0 and i_idx == 0:
                            x_lim, y_lim = (cur_x[:, 0].min(), cur_x[:, 0].max()), (cur_x[:, 1].min(), cur_x[:, 1].max())
                        axs[i_idx].set_xlim(x_lim)
                        axs[i_idx].set_ylim(y_lim)
                        axs[i_idx].scatter(cur_x[:, 0], cur_x[:, 1], s=10, label=labels[seq_idx])
                        axs[i_idx].set_title(f'$q(\\mathbf{{x}}_{{{config["n_steps"] - i}}})$')
                        if i_idx == 0:
                            axs[i_idx].legend()
                    
                    # Save final samples as CSV
                    if seq_idx == 0:
                        os.makedirs(f'{output_save_dir}/csv', exist_ok=True)
                        np.savetxt(f'{output_save_dir}/csv/step_{t}_{labels[seq_idx]}.csv', seq[-1].detach().cpu(), delimiter=',')
                    
                    # Create GIF if enabled
                    if config['create_gif']:
                        seq_stack = torch.stack(seq, dim=0).detach().cpu().numpy()
                        array_to_gif(seq_stack, f'{output_save_dir}/step_{t}_{labels[seq_idx]}.gif', 
                                     x_lim=x_lim, y_lim=y_lim, label=labels[seq_idx])
            
            plt.savefig(f'{output_save_dir}/step_{t}.png')
            plt.close(fig)
    
    # Save the final model
    save_model(model, config['name'], diff_dict, config['train_num_steps'], config['n_steps'], 
               config['dim'], config['model_pred_mode'], residual_func, ineq_func, opt_func)
    with open(f'{output_save_dir}/config.json', 'w') as f:
        json.dump(config, f)
    
    print(f"Completed training {model_name}. Model saved to {output_save_dir}")

# Combine final samples to create circular_samples.gif showing both models
phys_samples = np.loadtxt('./trained_models/toy/physics_model/csv/step_400_sample.csv', delimiter=',')
recon_samples = np.loadtxt('./trained_models/toy/recon_only/csv/step_400_sample.csv', delimiter=',')

plt.figure(figsize=(6, 6))
plt.scatter(phys_samples[:, 0], phys_samples[:, 1], s=10, color='purple', label='Physics-model')
plt.scatter(recon_samples[:, 0], recon_samples[:, 1], s=10, color='blue', label='Recon-only')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title('Toy hypersphere sampling')
plt.savefig('circular_samples.png')

print("\nTraining completed successfully!")
print("Now you can run sample_toy_guided.py to generate guided samples.")
print("Example command:")
print("python sample_toy_guided.py --physics_model_path ./trained_models/toy/physics_model --recon_model_path ./trained_models/toy/recon_only --physics_model_step 400 --recon_model_step 400 --guidance_scale 1.0 --n_samples 1000 --output_dir ./toy_guided_samples --create_gif") 