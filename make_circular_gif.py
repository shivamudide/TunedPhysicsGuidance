import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

# Colors for the two models
PHYSICS_COLOR = 'purple'
RECON_COLOR = 'blue'

def create_circular_gif():
    # Check if trained models exist
    phys_path = Path('./trained_models/toy/physics_model/csv/step_400_sample.csv')
    recon_path = Path('./trained_models/toy/recon_only/csv/step_400_sample.csv')
    
    if not phys_path.exists() or not recon_path.exists():
        print("Error: Model samples not found. Run train_toy_models.py first.")
        return
    
    # Load samples
    phys_samples = np.loadtxt(phys_path, delimiter=',')
    recon_samples = np.loadtxt(recon_path, delimiter=',')
    
    # Create static figure first
    plt.figure(figsize=(6, 6))
    plt.scatter(phys_samples[:, 0], phys_samples[:, 1], s=10, color=PHYSICS_COLOR, 
                label='Physics-model')
    plt.scatter(recon_samples[:, 0], recon_samples[:, 1], s=10, color=RECON_COLOR, 
                label='Recon-only')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Toy hypersphere sampling')
    plt.savefig('circular_samples.png')
    plt.close()
    
    print("Created circular_samples.png")
    
    # Now make an animated version that rotates
    # Determine bounds
    all_samples = np.vstack([phys_samples, recon_samples])
    x_min, x_max = all_samples[:, 0].min(), all_samples[:, 0].max()
    y_min, y_max = all_samples[:, 1].min(), all_samples[:, 1].max()
    
    # Add 10% padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)
    
    # Create GIF with rotation
    frames = []
    n_frames = 36  # For a full 360 degree rotation
    
    for i in range(n_frames):
        # Set up the plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        
        # Plot both sets of points
        ax.scatter(phys_samples[:, 0], phys_samples[:, 1], s=10, color=PHYSICS_COLOR, 
                   label='Physics-model')
        ax.scatter(recon_samples[:, 0], recon_samples[:, 1], s=10, color=RECON_COLOR, 
                   label='Recon-only')
        
        # Set the view angle - rotate around the z axis
        angle = i * (360 / n_frames)
        ax.view_init(30, angle)  # 30 degrees from xy-plane, rotate around z
        
        # Add labels and set limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.legend()
        ax.set_title('Toy hypersphere sampling')
        
        # Save to a temporary file
        temp_file = f'temp_frame_{i:03d}.png'
        plt.savefig(temp_file)
        plt.close()
        
        # Append to frames list
        frames.append(imageio.imread(temp_file))
    
    # Create the GIF
    imageio.mimsave('circular_samples.gif', frames, duration=0.1, loop=0)
    
    # Clean up temporary files
    for i in range(n_frames):
        os.remove(f'temp_frame_{i:03d}.png')
    
    print("Created circular_samples.gif")

if __name__ == '__main__':
    create_circular_gif() 