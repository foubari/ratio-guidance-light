"""
Generate and visualize samples from trained diffusion models.

Usage:
    # Generate from both models
    python src/visualize_diffusion_samples.py

    # Generate from specific model
    python src/visualize_diffusion_samples.py --dataset standard --num_samples 64

    # Save to specific location
    python src/visualize_diffusion_samples.py --output_dir my_samples
"""
import argparse
import torch
import torchvision.utils as vutils
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from models.unet import UNet
from utils.diffusion import DiffusionSchedule


def load_model(checkpoint_path, device):
    """Load a trained DDPM model."""
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('num_timesteps', 1000)


@torch.no_grad()
def generate_samples(model, schedule, num_samples, device):
    """Generate samples from a diffusion model."""
    print(f"Generating {num_samples} samples...")
    samples = schedule.sample(
        model=model,
        shape=(num_samples, 1, 28, 28),
        device=device
    )
    return samples


def visualize_comparison(standard_samples, rotated_samples, save_path):
    """Create a side-by-side comparison visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Convert tensors to grid images
    standard_grid = vutils.make_grid(standard_samples, nrow=4, normalize=True, value_range=(-1, 1))
    rotated_grid = vutils.make_grid(rotated_samples, nrow=4, normalize=True, value_range=(-1, 1))

    # Convert to numpy for display
    standard_np = standard_grid.permute(1, 2, 0).cpu().numpy()
    rotated_np = rotated_grid.permute(1, 2, 0).cpu().numpy()

    ax1.imshow(standard_np)
    ax1.set_title('Standard MNIST', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(rotated_np)
    ax2.set_title('Rotated MNIST (90°)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {save_path}")
    plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset is None:
        # Generate from both models
        print("\n" + "="*60)
        print("Generating samples from BOTH models")
        print("="*60)

        # Standard MNIST
        standard_ckpt = Path(args.checkpoint_dir) / 'standard' / 'best_model.pt'
        if not standard_ckpt.exists():
            print(f"Error: Standard model not found at {standard_ckpt}")
            print("Please train it first: python src/train_diffusion.py --dataset standard")
            return

        print("\nLoading standard MNIST model...")
        standard_model, num_timesteps = load_model(standard_ckpt, device)
        schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)

        print("Generating standard MNIST samples...")
        standard_samples = generate_samples(standard_model, schedule, args.num_samples, device)

        # Save standard samples
        vutils.save_image(
            standard_samples,
            output_dir / 'standard_samples.png',
            nrow=int(args.num_samples ** 0.5),
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"Saved standard samples to {output_dir / 'standard_samples.png'}")

        # Rotated MNIST
        rotated_ckpt = Path(args.checkpoint_dir) / 'rotated' / 'best_model.pt'
        if not rotated_ckpt.exists():
            print(f"\nWarning: Rotated model not found at {rotated_ckpt}")
            print("Skipping rotated samples. Train it with: python src/train_diffusion.py --dataset rotated")
        else:
            print("\nLoading rotated MNIST model...")
            rotated_model, num_timesteps = load_model(rotated_ckpt, device)
            schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)

            print("Generating rotated MNIST samples...")
            rotated_samples = generate_samples(rotated_model, schedule, args.num_samples, device)

            # Save rotated samples
            vutils.save_image(
                rotated_samples,
                output_dir / 'rotated_samples.png',
                nrow=int(args.num_samples ** 0.5),
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"Saved rotated samples to {output_dir / 'rotated_samples.png'}")

            # Create comparison visualization
            print("\nCreating comparison visualization...")
            visualize_comparison(
                standard_samples,
                rotated_samples,
                output_dir / 'comparison.png'
            )

    else:
        # Generate from single model
        print(f"\nGenerating samples from {args.dataset} model...")

        checkpoint_path = Path(args.checkpoint_dir) / args.dataset / 'best_model.pt'
        if not checkpoint_path.exists():
            print(f"Error: Model not found at {checkpoint_path}")
            print(f"Please train it first: python src/train_diffusion.py --dataset {args.dataset}")
            return

        model, num_timesteps = load_model(checkpoint_path, device)
        schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)

        samples = generate_samples(model, schedule, args.num_samples, device)

        # Save samples
        output_file = output_dir / f'{args.dataset}_samples.png'
        vutils.save_image(
            samples,
            output_file,
            nrow=int(args.num_samples ** 0.5),
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"Saved samples to {output_file}")

    print("\n" + "="*60)
    print("Done! Check the output directory:")
    print(f"  {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize samples from trained diffusion models')
    parser.add_argument('--dataset', type=str, choices=['standard', 'rotated'],
                       help='Which model to sample from (default: both)')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate (default: 16)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/diffusion',
                       help='Directory containing model checkpoints (default: checkpoints/diffusion)')
    parser.add_argument('--output_dir', type=str, default='outputs/diffusion_samples',
                       help='Directory to save samples (default: outputs/diffusion_samples)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    main(args)
