import argparse, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import subprocess, sys, shlex

sns.set_context('paper', font_scale=1.3)
sns.set_style('white')
plt.rcParams['pdf.fonttype'] = 42  # embed fonts for vector graphics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']


def load_image(path):
    """Load image as numpy array in [0,1]."""
    img = np.asarray(Image.open(path))
    if img.ndim == 2:
        img = img / 255.0
    else:
        img = img[..., :3] / 255.0  # drop alpha if present
    return img


def load_residual_mean(csv_path):
    if csv_path is None or not csv_path.exists():
        return None
    try:
        arr = np.loadtxt(csv_path, delimiter=',')
        return float(np.abs(arr).mean())
    except Exception:
        return None


def add_subplot(fig, ax, image, title, cmap=None):
    if image.ndim == 2:
        im = ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    else:
        im = ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    return im


def main():
    parser = argparse.ArgumentParser(description='Create a 4×3 summary figure for PIDM results')
    parser.add_argument('--base_dir', default='guided_samples/baseline/sample_0', help='Path to sample dir for weak baseline (guidance 0)')
    parser.add_argument('--guidance1_dir', default='guided_samples/guid1/sample_0', help='Path to sample dir for guidance=1 case')
    parser.add_argument('--best_dir', default='guided_samples/best_guidance/sample_0', help='Path to sample dir for best guidance before refinement/after')
    parser.add_argument('--refined_dir', default=None, help='Path to sample dir for after refinement (defaults to best_dir)')
    parser.add_argument('--auto', action='store_true', help='Automatically generate the samples before plotting')
    parser.add_argument('--output', default='summary_figure.pdf', help='Output file path (PDF)')
    args = parser.parse_args()

    if args.refined_dir is None:
        args.refined_dir = args.best_dir

    # ------------------------------------------------------------------
    # Optional: automatically run sample generation
    # ------------------------------------------------------------------
    if args.auto:
        base_root = Path('guided_samples')
        base_root.mkdir(exist_ok=True)

        runs = [
            {
                'name': 'baseline',
                'cmd': f"python sample_guided.py --physics_model_path ./trained_models/darcy/PIDM-ME --recon_model_path ./trained_models/recon_only --physics_model_step 300000 --recon_model_step 500 --guidance_scale 0.0 --n_samples 1 --output_dir ./guided_samples/baseline"
            },
            {
                'name': 'guid1',
                'cmd': f"python sample_guided.py --physics_model_path ./trained_models/darcy/PIDM-ME --recon_model_path ./trained_models/recon_only --physics_model_step 300000 --recon_model_step 10000 --guidance_scale 1.0 --n_samples 1 --output_dir ./guided_samples/guid1"
            },
            {
                'name': 'best_guidance',
                'cmd': f"python sample_guided.py --physics_model_path ./trained_models/darcy/PIDM-ME --recon_model_path ./trained_models/recon_only --physics_model_step 300000 --recon_model_step 10000 --guidance_scale 1.3 --schedule cosine --post_correction_iters 100 --n_samples 1 --extra_steps 200 --step_size 0.01 --smooth_weight 0.100 --output_dir ./guided_samples/best_guidance"
            }
        ]

        for run in runs:
            # Skip if results already exist
            out_dir = Path(f"guided_samples/{run['name']}/sample_0")
            if out_dir.exists() and any(out_dir.glob('field_0*.png')):
                print(f"[auto] Skipping {run['name']} — results already present.")
                continue
            print(f"[auto] Running: {run['cmd']}")
            try:
                subprocess.run(shlex.split(run['cmd']), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while running {run['name']}: {e}")
                sys.exit(1)

    # Each entry will hold (field0_path, field1_path, residual_path, label)
    rows = [
        (Path(args.base_dir), 'Weak model (checkpoint 500)'),
        (Path(args.guidance1_dir), 'Guidance = 1.0 (PDE + recon)'),
        (Path(args.best_dir), 'Best guidance (pre-refinement)'),
        (Path(args.refined_dir), 'After refinement')
    ]

    # Figure setup
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    col_titles = ['Pressure field', 'Permeability field', 'Residual magnitude']
    for i, ct in enumerate(col_titles):
        axes[0, i].set_title(ct, fontsize=12, pad=14)

    # Iterate rows
    for r, (sample_dir, row_label) in enumerate(rows):
        # Auto-detect sample_0 directory nested under physics_* if needed
        if not sample_dir.exists():
            nested_samples = list(sample_dir.parent.glob('**/sample_0'))
            sample_dir = nested_samples[0] if nested_samples else sample_dir

        # Determine image paths
        # Try _before or normal depending on dir
        def choose(path_glob):
            candidates = list(sample_dir.glob(path_glob))
            return candidates[0] if candidates else None

        # For rows 0–2 we may have "field_0_before"; for refined row maybe "field_0_after" or just field_0_after
        if 'refine' in row_label.lower() or 'after' in row_label.lower():
            suf = 'after'
        elif 'pre-refinement' in row_label.lower() or 'best' in row_label.lower():
            suf = 'before'
        else:
            suf = ''
        suffix = f'_{suf}.png' if suf else '.png'
        field0 = choose(f'field_0{suffix}') or choose('field_0.png')
        field1 = choose(f'field_1{suffix}') or choose('field_1.png')
        residual_img = choose('residual.png')
        residual_csv = choose('residual.csv')

        # Load images
        img0 = load_image(field0) if field0 else np.zeros((64, 64))
        img1 = load_image(field1) if field1 else np.zeros((64, 64))
        img_res = load_image(residual_img) if residual_img else np.zeros((64, 64))

        # Plot
        add_subplot(fig, axes[r, 0], img0, '' if r else col_titles[0])
        add_subplot(fig, axes[r, 1], img1, '' if r else col_titles[1])
        im_res = add_subplot(fig, axes[r, 2], img_res, '' if r else col_titles[2], cmap='plasma')

        # Row label on left side
        axes[r, 0].annotate(row_label, xy=(-0.25, 0.5), xycoords='axes fraction', rotation=90,
                            va='center', ha='center', fontsize=12, fontweight='bold')

        # Residual annotation
        res_mean = load_residual_mean(residual_csv)
        if res_mean is not None:
            txt = f"|R|₁ = {res_mean:6.2e}"
            axes[r, 2].text(0.97, 0.05, txt, transform=axes[r, 2].transAxes,
                           fontsize=8, color='white', ha='right', va='bottom',
                           bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none'))

    # Colorbar for residuals
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Residual (normalised)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {args.output}")


if __name__ == '__main__':
    main() 