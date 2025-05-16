import os
import argparse
import re
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Monitor training progress for diffusion models')
    parser.add_argument('--model_dir', type=str, default='./trained_models/recon_only',
                        help='Directory containing the model being trained')
    parser.add_argument('--total_iterations', type=int, default=300000,
                        help='Total training iterations expected')
    parser.add_argument('--update_interval', type=int, default=10,
                        help='Seconds between updates')
    parser.add_argument('--plot', action='store_true',
                        help='Show live plots')
    return parser.parse_args()

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the model directory"""
    checkpoint_pattern = re.compile(r'checkpoint_(\d+)\.pt$')
    checkpoint_files = []
    
    model_path = Path(model_dir) / 'model'
    if not model_path.exists():
        return None, 0
    
    for file in model_path.glob('checkpoint_*.pt'):
        match = checkpoint_pattern.search(file.name)
        if match:
            iteration = int(match.group(1))
            checkpoint_files.append((file, iteration))
    
    if not checkpoint_files:
        return None, 0
        
    checkpoint_files.sort(key=lambda x: x[1])
    return checkpoint_files[-1]

def estimate_completion_time(current_iteration, total_iterations, elapsed_seconds, iterations_history=None):
    """Estimate time to completion based on current progress and rate"""
    if current_iteration == 0:
        return "Unknown"
    
    # Calculate immediate rate (iterations per second)
    immediate_rate = current_iteration / elapsed_seconds
    
    # If we have history, use it to get a more stable rate
    if iterations_history and len(iterations_history) > 2:
        # Use the last 5 measurements at most
        history = iterations_history[-5:]
        iterations_diff = history[-1][0] - history[0][0]
        time_diff = history[-1][1] - history[0][1]
        if time_diff > 0:
            avg_rate = iterations_diff / time_diff
            # Blend rates to avoid wild fluctuations
            rate = (avg_rate * 0.7) + (immediate_rate * 0.3)
        else:
            rate = immediate_rate
    else:
        rate = immediate_rate
    
    # Calculate remaining time
    remaining_iterations = total_iterations - current_iteration
    remaining_seconds = remaining_iterations / rate if rate > 0 else float('inf')
    
    if remaining_seconds > 86400:  # More than a day
        days = remaining_seconds / 86400
        return f"{days:.1f} days"
    elif remaining_seconds > 3600:  # More than an hour
        hours = remaining_seconds / 3600
        return f"{hours:.1f} hours"
    else:
        minutes = remaining_seconds / 60
        return f"{minutes:.1f} minutes"

def parse_log_line(line):
    """Extract information from a log line"""
    # Look for common patterns in the output
    loss_match = re.search(r'training loss: ([\d\.e-]+)', line)
    if loss_match:
        return {'type': 'training', 'loss': float(loss_match.group(1))}
    
    test_match = re.search(r'test loss at iteration (\d+): ([\d\.e-]+)', line)
    if test_match:
        return {'type': 'test', 'iteration': int(test_match.group(1)), 'loss': float(test_match.group(2))}
    
    return None

def read_training_log(model_dir):
    """Try to find and read the training log file"""
    log_path = Path(model_dir) / 'training.log'
    
    if not log_path.exists():
        # Try to find any log files in the directory
        log_files = list(Path(model_dir).glob('*.log'))
        if not log_files:
            return []
        log_path = log_files[0]
    
    lines = []
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read log file: {e}")
    
    # Parse each line to extract information
    data = []
    for line in lines:
        result = parse_log_line(line)
        if result:
            data.append(result)
    
    return data

def format_time(seconds):
    """Format time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def update_progress_terminal(current_iteration, total_iterations, elapsed_time, estimated_time):
    """Update progress in terminal"""
    progress = current_iteration / total_iterations
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    elapsed_str = format_time(elapsed_time)
    
    print(f"\rProgress: [{bar}] {current_iteration}/{total_iterations} ({progress*100:.1f}%) - "
          f"Elapsed: {elapsed_str} - ETA: {estimated_time}", end='')

def monitor_training(args):
    """Monitor training progress and estimate completion time"""
    start_time = time.time()
    iterations_history = []
    training_loss_history = []
    test_loss_history = []
    
    # Setup plotting if enabled
    if args.plot:
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Training Progress')
        
        train_line, = ax1.plot([], [], 'b-', label='Training Loss')
        test_line, = ax1.plot([], [], 'r-', label='Test Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        
        progress_bar = ax2.barh([0], [0], color='blue')
        ax2.set_xlim(0, args.total_iterations)
        ax2.set_ylabel('Progress')
        ax2.set_yticks([])
        
        # Text annotations
        progress_text = ax2.text(10, 0, '', va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    print(f"Monitoring training in {args.model_dir}, refreshing every {args.update_interval} seconds...")
    print(f"Training target: {args.total_iterations} iterations")
    
    try:
        while True:
            # Find latest checkpoint
            latest_checkpoint, iteration = find_latest_checkpoint(args.model_dir)
            current_iteration = iteration
            
            # Calculate elapsed time and rate
            elapsed_time = time.time() - start_time
            iterations_history.append((current_iteration, elapsed_time))
            
            # Keep only the last 10 entries
            if len(iterations_history) > 10:
                iterations_history = iterations_history[-10:]
            
            # Estimate completion time
            eta = estimate_completion_time(current_iteration, args.total_iterations, elapsed_time, iterations_history)
            
            # Update terminal progress
            update_progress_terminal(current_iteration, args.total_iterations, elapsed_time, eta)
            
            # Try to read the log for loss information
            log_data = read_training_log(args.model_dir)
            
            if log_data:
                # Extract training and test losses
                for entry in log_data:
                    if entry['type'] == 'training':
                        training_loss_history.append((current_iteration, entry['loss']))
                    elif entry['type'] == 'test':
                        test_loss_history.append((entry['iteration'], entry['loss']))
            
            # Update plots if enabled
            if args.plot and training_loss_history:
                # Update loss plot
                train_iterations, train_losses = zip(*training_loss_history)
                train_line.set_data(train_iterations, train_losses)
                
                if test_loss_history:
                    test_iterations, test_losses = zip(*test_loss_history)
                    test_line.set_data(test_iterations, test_losses)
                
                # Update axis limits
                ax1.relim()
                ax1.autoscale_view()
                
                # Update progress bar
                progress_bar[0].set_width(current_iteration)
                completion_percentage = current_iteration / args.total_iterations * 100
                progress_text.set_text(f"{current_iteration}/{args.total_iterations} ({completion_percentage:.1f}%) - ETA: {eta}")
                progress_text.set_position((min(current_iteration + 10, args.total_iterations * 0.95), 0))
                
                plt.draw()
                plt.pause(0.1)
            
            # If training is complete
            if current_iteration >= args.total_iterations:
                print("\nTraining completed!")
                break
            
            time.sleep(args.update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    
    # Final status
    if current_iteration >= args.total_iterations:
        print(f"Training completed in {format_time(elapsed_time)}")
    else:
        progress = current_iteration / args.total_iterations
        print(f"Training progress: {current_iteration}/{args.total_iterations} ({progress*100:.1f}%)")
        print(f"Elapsed time: {format_time(elapsed_time)}")
        print(f"Estimated time to completion: {eta}")

if __name__ == "__main__":
    args = parse_args()
    monitor_training(args) 