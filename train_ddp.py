import os
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.data_utils import *
from src.denoising_utils import *
from src.unet_model import Unet3D
from src.residuals_darcy import ResidualsDarcy
from src.residuals_mechanics_K import ResidualsMechanics

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config_path):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)

    # Load config
    config = yaml.safe_load(Path(config_path).read_text())
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Model parameters
    gov_eqs = config['gov_eqs']
    fd_acc = config['fd_acc']
    diff_steps = config['diff_steps']
    use_dynamic_threshold = False
    self_condition = False
    use_double = False

    # Get dataset parameters based on problem type
    if gov_eqs == 'darcy':
        input_dim = 2
        output_dim = 2
        pixels_at_boundary = True
        domain_length = 1.
        reverse_d1 = True
        data_paths = ('./data/darcy/train/p_data.csv', './data/darcy/train/K_data.csv')
        data_paths_valid = ('./data/darcy/valid/p_data.csv', './data/darcy/valid/K_data.csv')
        bcs = 'none'
        pixels_per_dim = 64
        return_optimizer = False
        return_inequality = False
        ds = Dataset(data_paths, use_double=use_double)
        ds_valid = Dataset(data_paths_valid, use_double=use_double)
        train_batch_size = 32 if config['x0_estimation'] == 'mean' else 16
        sigmoid_last_channel = False
        train_iterations = 300000
    elif gov_eqs == 'mechanics':
        input_dim = 2
        output_dim = 3
        pixels_at_boundary = True
        reverse_d1 = True
        data_paths = ('./data/mechanics/train/fields/')
        data_paths_valid = ('./data/mechanics/test/valid/fields/')
        bcs = 'none'
        pixels_per_dim = 64
        return_optimizer = True
        return_inequality = True
        ds = Dataset_Paths(data_paths, use_double=use_double)
        ds_valid = Dataset_Paths(data_paths_valid, use_double=use_double)
        train_batch_size = 6 if config['x0_estimation'] == 'mean' else 4
        sigmoid_last_channel = True
        train_iterations = 600000

    # Create samplers for distributed training
    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(ds_valid, num_replicas=world_size, rank=rank)

    # Create dataloaders
    train_batch_size = train_batch_size // world_size  # Adjust batch size for distributed training
    dl = DataLoader(ds, batch_size=train_batch_size, sampler=train_sampler)
    dl_valid = DataLoader(ds_valid, batch_size=train_batch_size, sampler=valid_sampler)
    dl = cycle(dl)
    dl_valid = cycle(dl_valid)

    # Initialize diffusion utils
    diffusion_utils = DenoisingDiffusion(diff_steps, device, config['residual_grad_guidance'])

    # Create model
    if gov_eqs == 'darcy':
        model = Unet3D(dim=32, channels=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)
    elif gov_eqs == 'mechanics':
        model = Unet3D(dim=128, channels=output_dim+3+4, out_dim=output_dim, sigmoid_last_channel=sigmoid_last_channel).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Initialize residuals
    if gov_eqs == 'darcy':
        residuals = ResidualsDarcy(
            model=model, fd_acc=fd_acc, pixels_per_dim=pixels_per_dim,
            pixels_at_boundary=pixels_at_boundary, reverse_d1=reverse_d1,
            device=device, bcs=bcs, domain_length=domain_length,
            residual_grad_guidance=config['residual_grad_guidance'],
            use_ddim_x0=config['x0_estimation'] == 'sample',
            ddim_steps=config['ddim_steps']
        )
    elif gov_eqs == 'mechanics':
        residuals = ResidualsMechanics(
            model=model, pixels_per_dim=pixels_per_dim,
            pixels_at_boundary=pixels_at_boundary, device=device,
            bcs=bcs, no_BC_folder='./data/mechanics/solidspy_k_no_BC/',
            topopt_eval=True, use_ddim_x0=config['x0_estimation'] == 'sample',
            ddim_steps=config['ddim_steps']
        )

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1.e-4)

    # Create output directory
    if rank == 0:
        output_save_dir = f'./trained_models/{config["name"]}'
        os.makedirs(output_save_dir, exist_ok=True)

    # Training loop
    test_eval_freq = 500
    log_freq = 20
    ema_start = 1000
    ema = EMA(0.99)
    ema.register(model)

    pbar = tqdm(range(train_iterations+1)) if rank == 0 else range(train_iterations+1)
    for iteration in pbar:
        model.train()
        train_sampler.set_epoch(iteration)  # Important for proper shuffling
        
        cur_batch = next(dl).to(device)
        loss, data_loss, residual_loss, ineq_loss, opt_loss = diffusion_utils.model_estimation_loss(
            cur_batch, residual_func=residuals, c_data=config['c_data'],
            c_residual=config['c_residual'], c_ineq=config['c_ineq'],
            lambda_opt=config['lambda_opt']
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        if rank == 0:
            if iteration % log_freq == 0:
                if isinstance(pbar, tqdm):
                    pbar.set_description(f'training loss: {loss.item():.3e}')

            if iteration > ema_start:
                ema.update(model)

            # Evaluation on validation set
            if iteration % test_eval_freq == 0:
                model.eval()
                ema.ema(residuals.model)
                
                cur_test_batch = next(dl_valid).to(device)
                loss_test, data_loss_test, residual_loss_test, ineq_loss_test, opt_loss_test = diffusion_utils.model_estimation_loss(
                    cur_test_batch, residual_func=residuals,
                    c_data=config['c_data'], c_residual=config['c_residual'],
                    c_ineq=config['c_ineq'], lambda_opt=config['lambda_opt']
                )
                
                print(f'test loss at iteration {iteration}: {loss_test:.3e}')

                if iteration > 0:
                    save_model(config, model.module, iteration, output_save_dir)

            ema.restore(residuals.model)

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    config_path = 'model_recon_only.yaml'  # or pass as argument
    mp.spawn(
        train,
        args=(world_size, config_path),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 