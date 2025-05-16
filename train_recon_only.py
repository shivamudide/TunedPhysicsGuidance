import os, yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from src.data_utils import *
from torch.utils.data import DataLoader
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy

print("Starting script...")

# Load our reconstruction-only config
config_path = Path('model_recon_only.yaml')
if not config_path.exists():
    print(f"Error: Config file {config_path} not found.")
    print("Creating default config file...")
    default_config = {
        'name': 'recon_only',
        'gov_eqs': 'darcy',
        'fd_acc': 2,
        'diff_steps': 100,
        'x0_estimation': 'mean',
        'ddim_steps': 0,
        'residual_grad_guidance': False,
        'correction_mode': 'none',
        'M_correction': 0,
        'N_correction': 0,
        'c_data': 1.0,
        'c_residual': 0.0,
        'c_ineq': 0.0,
        'lambda_opt': 0.0
    }
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    config = default_config
else:
    config = yaml.safe_load(config_path.read_text())

name = config['name']
print(f"Using config: {name}")

# Training parameters
wandb_track = False  # Set to True to track training with wandb

# Get model parameters from config
fd_acc = config['fd_acc']
diff_steps = config['diff_steps']
c_data = config['c_data']
c_residual = config['c_residual']  # Should be 0.0
c_ineq = config['c_ineq']
lambda_opt = config['lambda_opt']
use_ddim_x0 = config['x0_estimation'] == 'sample'
ddim_steps = config['ddim_steps']
residual_grad_guidance = config['residual_grad_guidance']
correction_mode = config['correction_mode']
M_correction = config['M_correction']
N_correction = config['N_correction']
gov_eqs = config['gov_eqs']

# Other parameters
use_dynamic_threshold = False
self_condition = False
use_double = False

# Setup datasets - focusing on Darcy
input_dim = 2
output_dim = 2
pixels_at_boundary = True
domain_length = 1.
reverse_d1 = True
data_paths = ('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv')
data_paths_valid = ('./data/darcy/valid/p_data.csv', './data/darcy/valid/K_data.csv')

# Check if data files exist
for file_path in data_paths + data_paths_valid:
    if not os.path.exists(file_path):
        print(f"Error: Data file {file_path} not found.")
        print("Please make sure the data is correctly copied to the cloud machine.")
        exit(1)

print("All data files found. Setting up training...")

bcs = 'none'
pixels_per_dim = 64
return_optimizer = False
return_inequality = False

# Check if CUDA is available
print(f"CUDA availability: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

ds = Dataset(data_paths, use_double=use_double)
ds_valid = Dataset(data_paths_valid, use_double=use_double)
train_batch_size = 32 if not use_ddim_x0 else 16
sigmoid_last_channel = False
train_iterations = 10000  # reduced for recon-only fast sweep

if use_double:
    torch.set_default_dtype(torch.float64)

# Create data loaders
dl = cycle(DataLoader(ds, batch_size=train_batch_size, shuffle=False))
dl_valid = cycle(DataLoader(ds_valid, batch_size=train_batch_size, shuffle=False))

# Initialize diffusion utils
diffusion_utils = DenoisingDiffusion(diff_steps, device, residual_grad_guidance)

# Initialize model
model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)

# Initialize residuals calculator
residuals = ResidualsDarcy(
    model=model, fd_acc=fd_acc, pixels_per_dim=pixels_per_dim,
    pixels_at_boundary=pixels_at_boundary, reverse_d1=reverse_d1,
    device=device, bcs=bcs, domain_length=domain_length,
    residual_grad_guidance=residual_grad_guidance, 
    use_ddim_x0=use_ddim_x0, ddim_steps=ddim_steps
)

# Print model info
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=1.e-4)

# Initialize EMA
ema_start = 1000
ema = EMA(0.99)
ema.register(model)

# Set up tracking and logging
if wandb_track:
    import wandb
    wandb.init(project='pi_diffusion', name=name)
    log_fn = wandb.log
else:
    log_fn = noop
log_freq = 20

# Set up checkpoint directory
output_save_dir = f'./trained_models/{name}'
os.makedirs(os.path.join(output_save_dir, 'model'), exist_ok=True)
print(f"Saving checkpoints to: {output_save_dir}")

# Save config for this run
with open(os.path.join(output_save_dir, 'model', 'model.yaml'), 'w') as yaml_file:
    yaml.dump(dict(config), yaml_file, default_flow_style=False)

# Create evaluation parameters
test_eval_freq = 500
sample_freq = 20000
full_sample_freq = 100000

print("Starting training loop...")
# Training loop
pbar = tqdm(range(train_iterations+1))
for iteration in pbar:
    model.train()
    cur_batch = next(dl).to(device)
    
    # Compute loss - this will use c_residual=0.0 from config
    loss, data_loss, residual_loss, ineq_loss, opt_loss = diffusion_utils.model_estimation_loss(
                cur_batch, residual_func=residuals, c_data=c_data, c_residual=c_residual,
                c_ineq=c_ineq, lambda_opt=lambda_opt)    
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    
    # Logging
    if iteration % log_freq == 0:
        pbar.set_description(f'training loss: {loss.item():.3e}')
        log_fn({'loss': loss.item()}, step=iteration)
        log_fn({'loss_data': data_loss}, step=iteration)
        log_fn({'residual_mean_abs': residual_loss}, step=iteration)
    
    # EMA update
    if iteration > ema_start:
        ema.update(model)

    # Evaluation on validation set
    model.eval()
    ema.ema(residuals.model)
    if iteration % test_eval_freq == 0 and exists(dl_valid):
        cur_test_batch = next(dl_valid).to(device)
        
        loss_test, data_loss_test, residual_loss_test, ineq_loss_test, opt_loss_test = diffusion_utils.model_estimation_loss(
                    cur_test_batch, residual_func=residuals, c_data=c_data, c_residual=c_residual,
                    c_ineq=c_ineq, lambda_opt=lambda_opt)
        
        print(f'test loss at iteration {iteration}: {loss_test:.3e}')
        log_fn({'loss_test': loss_test.item()}, step=iteration)
        log_fn({'loss_data_test': data_loss_test}, step=iteration)
        log_fn({'residual_mean_abs_test': residual_loss_test}, step=iteration)

    # Save checkpoints at very early iterations and every 1000 until 10k
    early_saves = {1, 50, 100, 200, 500, 1000, 2000}

    if iteration in early_saves or (iteration % 1000 == 0 and iteration > 0):
        save_model(config, model, iteration, output_save_dir)
        print(f"Saved checkpoint at iteration {iteration}")

    ema.restore(residuals.model)

# Save final model
save_model(config, model, train_iterations, output_save_dir)
print("Training completed. Final model saved.") 