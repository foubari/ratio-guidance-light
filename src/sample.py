"""
Sample from DDPM with ratio-based guidance.

Usage:
    # Unconditional sampling
    python src/sample.py --num_samples 16 --dataset standard

    # Guided sampling
    python src/sample.py --num_samples 16 --dataset standard \
        --guided --loss_type disc --guidance_scale 2.0 \
        --condition_dataset rotated
"""
import argparse
import torch
import torchvision.utils as vutils
from pathlib import Path
import matplotlib.pyplot as plt

from models.unet import UNet
from models.ratio_estimator import RatioEstimator
from data.mnist_dataset import RotatedMNIST
from utils.diffusion import DiffusionSchedule
from utils.losses import DensityRatioLoss


def load_diffusion_model(checkpoint_path, device):
    """Load a trained DDPM model."""
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('num_timesteps', 1000)


def load_ratio_model(checkpoint_path, device):
    """Load a trained ratio estimator."""
    model = RatioEstimator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('loss_type', 'disc')


def get_guidance_function(ratio_model, condition_img, schedule, loss_type):
    """
    Create a guidance function for ratio-based sampling.

    Args:
        ratio_model: Trained RatioEstimator
        condition_img: Conditioning image (e.g., rotated MNIST) [B, 1, 28, 28]
        schedule: DiffusionSchedule
        loss_type: Type of loss used for training

    Returns:
        guidance_fn: Function that computes gradient for guidance
    """
    def guidance_fn(x_t, t):
        """
        Compute gradient of ratio score w.r.t. x_t.

        Args:
            x_t: Noisy image [B, 1, 28, 28]
            t: Timesteps [B]

        Returns:
            gradient: Gradient of score [B, 1, 28, 28]
        """
        # Add noise to condition image at the same timestep
        condition_noisy, _ = schedule.add_noise(condition_img, t)

        # Compute score
        with torch.enable_grad():
            x_t_grad = x_t.requires_grad_(True)
            scores = ratio_model(x_t_grad, condition_noisy, t)

            # For discriminator and DV, the score is already log-ratio-like
            # For uLSIF/RuLSIF/KLIEP, we need to map to ratio first
            if loss_type in ['ulsif', 'kliep']:
                # w = softplus(T)
                scores = torch.nn.functional.softplus(scores)
                # We want gradient of log(w)
                scores = torch.log(scores + 1e-8)
            elif loss_type == 'rulsif':
                # For rulsif, similar mapping
                scores = torch.nn.functional.softplus(scores)
                scores = torch.log(scores + 1e-8)

            # Sum over batch (we want scalar for backward)
            score_sum = scores.sum()

            # Compute gradient
            gradient = torch.autograd.grad(score_sum, x_t_grad)[0]

        return gradient

    return guidance_fn


@torch.no_grad()
def sample_unconditional(
    model,
    schedule,
    num_samples=16,
    device='cuda',
    save_path='outputs/unconditional.png'
):
    """Sample unconditionally from DDPM."""
    print(f"Generating {num_samples} unconditional samples...")

    # Sample
    samples = schedule.sample(
        model=model,
        shape=(num_samples, 1, 28, 28),
        device=device
    )

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        samples,
        save_path,
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"Saved samples to {save_path}")

    return samples


@torch.no_grad()
def sample_guided(
    model,
    ratio_model,
    schedule,
    condition_dataset,
    loss_type='disc',
    num_samples=16,
    guidance_scale=2.0,
    device='cuda',
    save_path='outputs/guided.png'
):
    """
    Sample with ratio-based guidance.

    Args:
        model: DDPM model
        ratio_model: Ratio estimator
        schedule: Diffusion schedule
        condition_dataset: Dataset to sample conditioning images from
        loss_type: Loss type used for training ratio model
        num_samples: Number of samples to generate
        guidance_scale: Guidance strength
        device: Device
        save_path: Path to save results
    """
    print(f"Generating {num_samples} guided samples (scale={guidance_scale})...")

    # Sample conditioning images
    indices = torch.randperm(len(condition_dataset))[:num_samples]
    condition_imgs = []
    for idx in indices:
        img, _ = condition_dataset[int(idx)]
        condition_imgs.append(img)
    condition_imgs = torch.stack(condition_imgs).to(device)

    # Create guidance function
    guidance_fn = get_guidance_function(ratio_model, condition_imgs, schedule, loss_type)

    # Sample with guidance
    samples = schedule.sample(
        model=model,
        shape=(num_samples, 1, 28, 28),
        guidance_fn=guidance_fn,
        guidance_scale=guidance_scale,
        device=device
    )

    # Save paired visualization
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create grid with condition and generated side by side
    paired = torch.stack([condition_imgs, samples], dim=1)  # [B, 2, 1, 28, 28]
    paired = paired.view(-1, 1, 28, 28)  # [B*2, 1, 28, 28]

    vutils.save_image(
        paired,
        save_path,
        nrow=8,  # 4 pairs per row
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"Saved paired samples to {save_path}")

    # Also save separately
    save_dir = save_path.parent
    vutils.save_image(
        condition_imgs,
        save_dir / 'conditions.png',
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )
    vutils.save_image(
        samples,
        save_dir / 'generated.png',
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    return samples, condition_imgs


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load diffusion model
    diffusion_ckpt = Path(args.diffusion_checkpoint) / args.dataset / 'best_model.pt'
    if not diffusion_ckpt.exists():
        raise FileNotFoundError(f"Diffusion model not found: {diffusion_ckpt}")

    model, num_timesteps = load_diffusion_model(diffusion_ckpt, device)
    schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)
    print(f"Loaded diffusion model from {diffusion_ckpt}")

    if args.guided:
        # Load ratio model
        ratio_ckpt = Path(args.ratio_checkpoint) / args.loss_type / 'best_model.pt'
        if not ratio_ckpt.exists():
            raise FileNotFoundError(f"Ratio model not found: {ratio_ckpt}")

        ratio_model, loss_type = load_ratio_model(ratio_ckpt, device)
        print(f"Loaded ratio model from {ratio_ckpt} (loss_type: {loss_type})")

        # Load conditioning dataset
        condition_dataset = RotatedMNIST(
            root='./data',
            train=False,
            rotate=(args.condition_dataset == 'rotated'),
            download=True
        )

        # Sample with guidance
        save_path = f'outputs/guided_{args.dataset}_{args.loss_type}_scale{args.guidance_scale}.png'
        sample_guided(
            model=model,
            ratio_model=ratio_model,
            schedule=schedule,
            condition_dataset=condition_dataset,
            loss_type=loss_type,
            num_samples=args.num_samples,
            guidance_scale=args.guidance_scale,
            device=device,
            save_path=save_path
        )
    else:
        # Unconditional sampling
        save_path = f'outputs/unconditional_{args.dataset}.png'
        sample_unconditional(
            model=model,
            schedule=schedule,
            num_samples=args.num_samples,
            device=device,
            save_path=save_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from DDPM')
    parser.add_argument('--dataset', type=str, choices=['standard', 'rotated'], required=True,
                       help='Which diffusion model to use')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate (default: 16)')
    parser.add_argument('--guided', action='store_true',
                       help='Use ratio-based guidance')
    parser.add_argument('--loss_type', type=str,
                       choices=['disc', 'dv', 'ulsif', 'rulsif', 'kliep'],
                       default='disc',
                       help='Loss type for ratio model (default: disc)')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                       help='Guidance scale (default: 2.0)')
    parser.add_argument('--condition_dataset', type=str,
                       choices=['standard', 'rotated'],
                       default='rotated',
                       help='Dataset for conditioning images (default: rotated)')
    parser.add_argument('--diffusion_checkpoint', type=str, default='checkpoints/diffusion',
                       help='Path to diffusion checkpoints (default: checkpoints/diffusion)')
    parser.add_argument('--ratio_checkpoint', type=str, default='checkpoints/ratio',
                       help='Path to ratio checkpoints (default: checkpoints/ratio)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    main(args)
