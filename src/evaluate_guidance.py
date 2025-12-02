"""
Evaluate guided sampling quality by computing label matching accuracy.

This script:
1. Loads trained classifiers for standard and rotated MNIST
2. Loads trained diffusion models and ratio estimators
3. Generates guided samples (standard conditioned on rotated)
4. Computes matching accuracy: % of pairs with same predicted label

Usage:
    # Evaluate specific loss type
    python src/evaluate_guidance.py --loss_type disc --guidance_scale 2.0

    # Evaluate all loss types
    python src/evaluate_guidance.py --eval_all

    # Evaluate multiple guidance scales
    python src/evaluate_guidance.py --loss_type disc --scales 0.5 1.0 2.0 5.0
"""
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
import torchvision.utils as vutils

from models.unet import UNet
from models.ratio_estimator import RatioEstimator
from models.classifier import MNISTClassifier
from data.mnist_dataset import RotatedMNIST
from utils.diffusion import DiffusionSchedule
from utils.losses import DensityRatioLoss
from utils.path_utils import get_checkpoint_path, list_available_models


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


def load_classifier(checkpoint_path, device):
    """Load a trained classifier."""
    model = MNISTClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_guidance_function(ratio_model, condition_img, schedule, loss_type):
    """Create a guidance function for ratio-based sampling."""
    def guidance_fn(x_t, t):
        # Add noise to condition image at the same timestep
        condition_noisy, _ = schedule.add_noise(condition_img, t)

        # Compute score
        with torch.enable_grad():
            x_t_grad = x_t.requires_grad_(True)
            scores = ratio_model(x_t_grad, condition_noisy, t)

            # Map to log-ratio for guidance
            if loss_type in ['ulsif', 'kliep']:
                scores = torch.nn.functional.softplus(scores)
                scores = torch.log(scores + 1e-8)
            elif loss_type == 'rulsif':
                scores = torch.nn.functional.softplus(scores)
                scores = torch.log(scores + 1e-8)

            score_sum = scores.sum()
            gradient = torch.autograd.grad(score_sum, x_t_grad)[0]

        return gradient

    return guidance_fn


@torch.no_grad()
def evaluate_guidance(
    loss_type='disc',
    guidance_scale=2.0,
    num_samples=100,
    device='cuda',
    diffusion_checkpoint_dir='checkpoints/diffusion',
    ratio_checkpoint_dir='checkpoints/ratio',
    classifier_checkpoint_dir='checkpoints/classifiers',
    save_results=True,
    output_dir='outputs/evaluation',
    # Hyperparameters for finding the correct checkpoint
    rulsif_alpha=None,
    rulsif_link=None,
    kliep_lambda=None,
    infonce_tau=None,
    ulsif_l2=None,
    alpha_div_alpha=None,
    mine_use_ema=None,
    mine_ema_rate=None
):
    """
    Evaluate guided sampling accuracy.

    Returns:
        dict: Results with matching accuracy and predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {loss_type} loss, guidance_scale={guidance_scale}")
    print(f"{'='*60}\n")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load diffusion model (standard MNIST)
    diffusion_ckpt = Path(diffusion_checkpoint_dir) / 'standard' / 'best_model.pt'
    if not diffusion_ckpt.exists():
        raise FileNotFoundError(f"Diffusion model not found: {diffusion_ckpt}")

    print("Loading diffusion model...")
    model, num_timesteps = load_diffusion_model(diffusion_ckpt, device)
    schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)

    # Load ratio model with hyperparameter-based path
    # Use defaults if not specified
    hyperparams = {
        'rulsif_alpha': rulsif_alpha if rulsif_alpha is not None else 0.2,
        'rulsif_link': rulsif_link if rulsif_link is not None else 'exp',
        'kliep_lambda': kliep_lambda if kliep_lambda is not None else 1.0,
        'infonce_tau': infonce_tau if infonce_tau is not None else 0.07,
        'ulsif_l2': ulsif_l2 if ulsif_l2 is not None else 0.0,
        'alpha_div_alpha': alpha_div_alpha if alpha_div_alpha is not None else 0.5,
        'mine_use_ema': mine_use_ema if mine_use_ema is not None else True,
        'mine_ema_rate': mine_ema_rate if mine_ema_rate is not None else 0.99,
    }

    ratio_ckpt_dir = get_checkpoint_path(ratio_checkpoint_dir, loss_type, **hyperparams)
    ratio_ckpt = ratio_ckpt_dir / 'best_model.pt'

    if not ratio_ckpt.exists():
        # Try to find an available model if exact match not found
        available = list_available_models(ratio_checkpoint_dir, loss_type)
        if available:
            print(f"⚠️  Model not found at: {ratio_ckpt}")
            print(f"Available {loss_type} models:")
            for model_info in available:
                print(f"  - {model_info['path'].name}")
            raise FileNotFoundError(
                f"Ratio model not found: {ratio_ckpt}\n"
                f"Use hyperparameter flags (--rulsif_link, --kliep_lambda, etc.) to specify the model."
            )
        else:
            raise FileNotFoundError(f"No {loss_type} models found in {ratio_checkpoint_dir}")

    print(f"Loading ratio model ({loss_type})...")
    print(f"  Path: {ratio_ckpt_dir.name}")
    ratio_model, _ = load_ratio_model(ratio_ckpt, device)

    # Load classifiers
    standard_classifier_ckpt = Path(classifier_checkpoint_dir) / 'standard_classifier.pt'
    rotated_classifier_ckpt = Path(classifier_checkpoint_dir) / 'rotated_classifier.pt'

    if not standard_classifier_ckpt.exists() or not rotated_classifier_ckpt.exists():
        raise FileNotFoundError(
            f"Classifiers not found. Please train them first:\n"
            f"  python src/train_classifier.py --dataset both"
        )

    print("Loading classifiers...")
    standard_classifier = load_classifier(standard_classifier_ckpt, device)
    rotated_classifier = load_classifier(rotated_classifier_ckpt, device)

    # Load conditioning dataset (rotated MNIST test set)
    print("Loading condition dataset...")
    condition_dataset = RotatedMNIST(root='./data', train=False, rotate=True, download=True)

    # Generate guided samples in batches
    print(f"\nGenerating {num_samples} guided samples...")
    batch_size = 16
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_condition_imgs = []
    all_generated_imgs = []
    all_condition_labels = []
    all_generated_labels = []

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Get batch of condition images
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_actual_size = end_idx - start_idx

        condition_imgs = []
        for i in range(start_idx, end_idx):
            img, _ = condition_dataset[i]
            condition_imgs.append(img)
        condition_imgs = torch.stack(condition_imgs).to(device)

        # Create guidance function
        guidance_fn = get_guidance_function(ratio_model, condition_imgs, schedule, loss_type)

        # Sample with guidance
        samples = schedule.sample(
            model=model,
            shape=(batch_actual_size, 1, 28, 28),
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
            device=device
        )

        # Classify
        with torch.no_grad():
            condition_preds = rotated_classifier.predict(condition_imgs)
            generated_preds = standard_classifier.predict(samples)

        # Store
        all_condition_imgs.append(condition_imgs.cpu())
        all_generated_imgs.append(samples.cpu())
        all_condition_labels.append(condition_preds.cpu())
        all_generated_labels.append(generated_preds.cpu())

    # Concatenate all results
    condition_imgs_all = torch.cat(all_condition_imgs, dim=0)
    generated_imgs_all = torch.cat(all_generated_imgs, dim=0)
    condition_labels_all = torch.cat(all_condition_labels, dim=0)
    generated_labels_all = torch.cat(all_generated_labels, dim=0)

    # Compute matching accuracy
    matches = (condition_labels_all == generated_labels_all).float()
    accuracy = matches.mean().item() * 100.0

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Matching Accuracy: {accuracy:.2f}%")
    print(f"  Matches: {matches.sum().int()}/{len(matches)}")
    print(f"{'='*60}\n")

    # Save results
    results = {
        'loss_type': loss_type,
        'guidance_scale': guidance_scale,
        'num_samples': num_samples,
        'accuracy': accuracy,
        'num_matches': matches.sum().item(),
        'total_samples': len(matches),
    }

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_file = output_path / f'{loss_type}_scale{guidance_scale}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")

        # Save sample visualizations (first 16)
        vis_samples = min(16, num_samples)
        paired = torch.stack([
            condition_imgs_all[:vis_samples],
            generated_imgs_all[:vis_samples]
        ], dim=1).view(-1, 1, 28, 28)

        vis_file = output_path / f'{loss_type}_scale{guidance_scale}_samples.png'
        vutils.save_image(
            paired,
            vis_file,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"Saved visualizations to {vis_file}")

    return results


def evaluate_multiple_scales(loss_type, scales, **kwargs):
    """Evaluate a loss type across multiple guidance scales."""
    results_list = []

    for scale in scales:
        results = evaluate_guidance(
            loss_type=loss_type,
            guidance_scale=scale,
            **kwargs
        )
        results_list.append(results)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for {loss_type}:")
    print(f"{'='*60}")
    for res in results_list:
        print(f"  Scale {res['guidance_scale']:4.1f}: {res['accuracy']:5.2f}% accuracy")
    print(f"{'='*60}\n")

    return results_list


def evaluate_all_losses(guidance_scale, **kwargs):
    """Evaluate all available loss types."""
    loss_types = ['disc', 'dv', 'ulsif', 'rulsif', 'kliep']
    available_losses = []

    # Check which ratio models are available
    ratio_dir = Path(kwargs.get('ratio_checkpoint_dir', 'checkpoints/ratio'))
    for loss_type in loss_types:
        if (ratio_dir / loss_type / 'best_model.pt').exists():
            available_losses.append(loss_type)

    if not available_losses:
        print("No trained ratio models found!")
        return []

    print(f"Found trained models: {available_losses}")

    results_list = []
    for loss_type in available_losses:
        try:
            results = evaluate_guidance(
                loss_type=loss_type,
                guidance_scale=guidance_scale,
                **kwargs
            )
            results_list.append(results)
        except Exception as e:
            print(f"Error evaluating {loss_type}: {e}")

    # Print comparison
    print(f"\n{'='*60}")
    print(f"Comparison at guidance_scale={guidance_scale}:")
    print(f"{'='*60}")
    for res in sorted(results_list, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {res['loss_type']:8s}: {res['accuracy']:5.2f}% accuracy")
    print(f"{'='*60}\n")

    return results_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate guided sampling')
    parser.add_argument('--loss_type', type=str,
                       choices=['disc', 'dv', 'ulsif', 'rulsif', 'kliep', 'infonce', 'nce', 'alpha_div', 'mine', 'ctsm', 'ctsm_v'],
                       help='Loss type to evaluate')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                       help='Guidance scale (default: 2.0)')
    parser.add_argument('--scales', type=float, nargs='+',
                       help='Multiple guidance scales to evaluate')
    parser.add_argument('--eval_all', action='store_true',
                       help='Evaluate all available loss types')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory (default: outputs/evaluation)')

    # Hyperparameter arguments for loading specific models
    parser.add_argument('--rulsif_alpha', type=float, default=None,
                       help='RuLSIF alpha parameter (default: 0.2 if not specified)')
    parser.add_argument('--rulsif_link', type=str, default=None,
                       choices=['exp', 'softplus', 'identity'],
                       help='RuLSIF link function (default: exp if not specified)')
    parser.add_argument('--kliep_lambda', type=float, default=None,
                       help='KLIEP lambda parameter (default: 1.0 if not specified)')
    parser.add_argument('--infonce_tau', type=float, default=None,
                       help='InfoNCE temperature (default: 0.07 if not specified)')
    parser.add_argument('--ulsif_l2', type=float, default=None,
                       help='uLSIF L2 regularization (default: 0.0 if not specified)')
    parser.add_argument('--alpha_div_alpha', type=float, default=None,
                       help='α-Divergence alpha parameter (default: 0.5 if not specified)')
    parser.add_argument('--mine_use_ema', type=bool, default=None,
                       help='MINE use EMA (default: True if not specified)')
    parser.add_argument('--mine_ema_rate', type=float, default=None,
                       help='MINE EMA rate (default: 0.99 if not specified)')

    args = parser.parse_args()

    # Determine what to evaluate
    if args.eval_all:
        evaluate_all_losses(
            guidance_scale=args.guidance_scale,
            num_samples=args.num_samples,
            device=args.device,
            output_dir=args.output_dir
        )
    elif args.scales:
        if not args.loss_type:
            print("Error: --loss_type required when using --scales")
        else:
            evaluate_multiple_scales(
                loss_type=args.loss_type,
                scales=args.scales,
                num_samples=args.num_samples,
                device=args.device,
                output_dir=args.output_dir
            )
    else:
        if not args.loss_type:
            print("Error: --loss_type required (or use --eval_all)")
        else:
            evaluate_guidance(
                loss_type=args.loss_type,
                guidance_scale=args.guidance_scale,
                num_samples=args.num_samples,
                device=args.device,
                output_dir=args.output_dir,
                rulsif_alpha=args.rulsif_alpha,
                rulsif_link=args.rulsif_link,
                kliep_lambda=args.kliep_lambda,
                infonce_tau=args.infonce_tau,
                ulsif_l2=args.ulsif_l2,
                alpha_div_alpha=args.alpha_div_alpha,
                mine_use_ema=args.mine_use_ema,
                mine_ema_rate=args.mine_ema_rate
            )
