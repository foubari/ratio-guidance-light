"""
Train CTSM/CTSM-v time score estimators for density ratio estimation.

Usage:
    python src/train_ctsm.py --loss_type ctsm --epochs 30
    python src/train_ctsm.py --loss_type ctsm_v --epochs 30
"""
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models.ratio_estimator import TimeScoreEstimator, VectorizedTimeScoreEstimator
from data.mnist_dataset import RatioTrainingDataset
from utils.ctsm_losses import (
    ctsm_loss, ctsm_v_loss,
    compute_conditional_time_score_vp,
    compute_vectorized_conditional_time_score_vp
)
from utils.path_utils import get_checkpoint_path


def train_epoch_ctsm(model, dataloader, optimizer, device, weighting='time_score_norm'):
    """
    Train one epoch for CTSM (scalar time score).

    CTSM learns the time score for both:
    - p_1 (joint distribution): real pairs (x_real, y_real)
    - p_0 (product of marginals): fake pairs (x_fake, y_fake)

    The model learns to estimate ∂_t log p_t(x,y) for both distributions.
    """
    model.train()
    total_loss = 0
    all_metrics = []

    for batch_idx, batch in enumerate(dataloader):
        # Extract data from batch dictionary
        x_img = batch['img1'].to(device)  # standard MNIST
        y_img = batch['img2'].to(device)  # rotated MNIST
        is_real = batch['is_real'].to(device)

        # Separate real and fake pairs
        real_mask = is_real > 0.5
        fake_mask = ~real_mask

        # Skip if we don't have both real and fake samples
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            continue

        x_real = x_img[real_mask]
        y_real = y_img[real_mask]
        x_fake = x_img[fake_mask]
        y_fake = y_img[fake_mask]

        # Combine real and fake pairs for batch processing
        x_combined = torch.cat([x_real, x_fake], dim=0)
        y_combined = torch.cat([y_real, y_fake], dim=0)
        batch_size = x_combined.shape[0]

        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)

        # Create noisy samples following VP path
        # x_t = α_t·x_1 + √(1-α_t²)·ε, where x_1 = clean image, ε ~ N(0, I)
        alpha_t = t.view(-1, 1, 1, 1)
        noise_x = torch.randn_like(x_combined)
        noise_y = torch.randn_like(y_combined)

        x_noisy = alpha_t * x_combined + torch.sqrt(1 - alpha_t**2 + 1e-8) * noise_x
        y_noisy = alpha_t * y_combined + torch.sqrt(1 - alpha_t**2 + 1e-8) * noise_y

        # Compute target time score in closed form (no learning!)
        # This computes ∂_t log p_t(x,y|z) where z is the clean data
        target_time_score = compute_conditional_time_score_vp(
            x_noisy, x_combined, t, alpha_schedule='linear'
        )

        # Forward pass: predict time score
        pred_time_score = model(x_noisy, y_noisy, t)

        # Compute CTSM loss
        loss, metrics = ctsm_loss(
            pred_time_score, target_time_score, t,
            weighting=weighting, alpha_schedule='linear'
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_metrics.append(metrics)

    # Average metrics
    if len(all_metrics) == 0:
        return {'loss': float('inf')}

    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(all_metrics)

    return avg_metrics


def train_epoch_ctsm_v(model, dataloader, optimizer, device, weighting='time_score_norm'):
    """
    Train one epoch for CTSM-v (vectorized time score).

    CTSM-v learns the vectorized time score for both:
    - p_1 (joint distribution): real pairs (x_real, y_real)
    - p_0 (product of marginals): fake pairs (x_fake, y_fake)

    The model learns to estimate ∂_t log p_t(x,y) as a D-dimensional vector.
    """
    model.train()
    total_loss = 0
    all_metrics = []

    for batch_idx, batch in enumerate(dataloader):
        # Extract data from batch dictionary
        x_img = batch['img1'].to(device)  # standard MNIST
        y_img = batch['img2'].to(device)  # rotated MNIST
        is_real = batch['is_real'].to(device)

        # Separate real and fake pairs
        real_mask = is_real > 0.5
        fake_mask = ~real_mask

        # Skip if we don't have both real and fake samples
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            continue

        x_real = x_img[real_mask]
        y_real = y_img[real_mask]
        x_fake = x_img[fake_mask]
        y_fake = y_img[fake_mask]

        # Combine real and fake pairs for batch processing
        x_combined = torch.cat([x_real, x_fake], dim=0)
        y_combined = torch.cat([y_real, y_fake], dim=0)
        batch_size = x_combined.shape[0]

        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)

        # Create noisy samples following VP path
        alpha_t = t.view(-1, 1, 1, 1)
        noise_x = torch.randn_like(x_combined)
        noise_y = torch.randn_like(y_combined)

        x_noisy = alpha_t * x_combined + torch.sqrt(1 - alpha_t**2 + 1e-8) * noise_x
        y_noisy = alpha_t * y_combined + torch.sqrt(1 - alpha_t**2 + 1e-8) * noise_y

        # Compute target vectorized time score in closed form
        # This computes ∂_t log p_t(x,y|z) where z is the clean data
        target_vec_time_score = compute_vectorized_conditional_time_score_vp(
            x_noisy, x_combined, t, alpha_schedule='linear'
        )

        # Forward pass: predict vectorized time score
        pred_vec_time_score = model(x_noisy, y_noisy, t)

        # Compute CTSM-v loss
        loss, metrics = ctsm_v_loss(
            pred_vec_time_score, target_vec_time_score, t,
            weighting=weighting, alpha_schedule='linear'
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_metrics.append(metrics)

    # Average metrics
    if len(all_metrics) == 0:
        return {'loss': float('inf')}

    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(all_metrics)

    return avg_metrics


def validate_ctsm(model, dataloader, device, weighting='time_score_norm', is_vectorized=False):
    """
    Validate CTSM/CTSM-v model.
    """
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for batch in dataloader:
            # Extract data from batch dictionary
            x_img = batch['img1'].to(device)
            y_img = batch['img2'].to(device)
            is_real = batch['is_real'].to(device)

            # Separate real and fake pairs
            real_mask = is_real > 0.5
            fake_mask = ~real_mask

            # Skip if we don't have both real and fake samples
            if real_mask.sum() == 0 or fake_mask.sum() == 0:
                continue

            x_real = x_img[real_mask]
            y_real = y_img[real_mask]
            x_fake = x_img[fake_mask]
            y_fake = y_img[fake_mask]

            # Combine real and fake pairs
            x_combined = torch.cat([x_real, x_fake], dim=0)
            y_combined = torch.cat([y_real, y_fake], dim=0)
            batch_size = x_combined.shape[0]

            # Sample time
            t = torch.rand(batch_size, device=device)

            # Create noisy samples
            alpha_t = t.view(-1, 1, 1, 1)
            noise_x = torch.randn_like(x_combined)
            noise_y = torch.randn_like(y_combined)

            x_noisy = alpha_t * x_combined + torch.sqrt(1 - alpha_t**2 + 1e-8) * noise_x
            y_noisy = alpha_t * y_combined + torch.sqrt(1 - alpha_t**2 + 1e-8) * noise_y

            # Compute target and prediction
            if is_vectorized:
                target = compute_vectorized_conditional_time_score_vp(x_noisy, x_combined, t)
                pred = model(x_noisy, y_noisy, t)
                loss, metrics = ctsm_v_loss(pred, target, t, weighting=weighting)
            else:
                target = compute_conditional_time_score_vp(x_noisy, x_combined, t)
                pred = model(x_noisy, y_noisy, t)
                loss, metrics = ctsm_loss(pred, target, t, weighting=weighting)

            total_loss += loss.item()
            all_metrics.append(metrics)

    # Average metrics
    if len(all_metrics) == 0:
        return {'loss': float('inf')}

    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(all_metrics)

    return avg_metrics


def train_ctsm_estimator(
    loss_type='ctsm',  # 'ctsm' or 'ctsm_v'
    epochs=30,
    batch_size=256,  # CTSM benefits from larger batches
    lr=2e-3,  # Higher LR than standard methods
    real_fake_ratio=0.5,  # CTSM needs both real (p_1) and fake (p_0) pairs
    num_timesteps=1000,
    device='cuda',
    save_dir='checkpoints/ratio',
    num_workers=4,
    patience=10,
    resume_from=None,
    weighting='time_score_norm',  # 'time_score_norm', 'stein_score_norm', or 'uniform'
):
    """
    Train CTSM or CTSM-v time score estimator.

    Args:
        loss_type: 'ctsm' (scalar) or 'ctsm_v' (vectorized)
        epochs: Number of training epochs
        batch_size: Batch size (256-512 recommended)
        lr: Learning rate (1e-3 to 2e-3 recommended)
        real_fake_ratio: Ratio of real (joint p_1) to fake (product p_0) pairs (default: 0.5)
        weighting: Weighting function for loss ('time_score_norm', 'stein_score_norm', or 'uniform')
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Hyperparameters for path generation
    hyperparams = {
        'weighting': weighting,
    }

    # Create save directory
    save_path = get_checkpoint_path(save_dir, loss_type, **hyperparams)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {loss_type.upper()} Time Score Estimator")
    print(f"{'='*60}")
    print(f"Save path: {save_path}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Weighting: {weighting}")

    # Model
    if loss_type == 'ctsm':
        model = TimeScoreEstimator().to(device)
        is_vectorized = False
    elif loss_type == 'ctsm_v':
        model = VectorizedTimeScoreEstimator().to(device)
        is_vectorized = True
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'ctsm' or 'ctsm_v'")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (higher LR for CTSM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Data
    train_dataset = RatioTrainingDataset(
        train=True,
        real_fake_ratio=1.0,  # CTSM uses only real pairs
    )
    val_dataset = RatioTrainingDataset(
        train=False,
        real_fake_ratio=1.0,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from:
        checkpoint_path = save_path / resume_from
        if checkpoint_path.exists():
            print(f"\nResuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            print(f"Resuming from epoch {start_epoch}")

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        # Train
        if loss_type == 'ctsm':
            train_metrics = train_epoch_ctsm(model, train_loader, optimizer, device, weighting)
        else:
            train_metrics = train_epoch_ctsm_v(model, train_loader, optimizer, device, weighting)

        # Validate
        val_metrics = validate_ctsm(model, val_loader, device, weighting, is_vectorized)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Print metrics
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'  Train Loss: {train_metrics["loss"]:.4f}')
        print(f'  Val Loss: {val_metrics["loss"]:.4f}')

        if loss_type == 'ctsm':
            print(f'  Train MSE: {train_metrics.get("mse", 0):.4f}')
            print(f'  Val MSE: {val_metrics.get("mse", 0):.4f}')
            print(f'  Target mean: {val_metrics.get("target_time_score_mean", 0):.4f}')
            print(f'  Pred mean: {val_metrics.get("pred_time_score_mean", 0):.4f}')
        else:
            print(f'  Train MSE per dim: {train_metrics.get("mse_per_dim", 0):.6f}')
            print(f'  Val MSE per dim: {val_metrics.get("mse_per_dim", 0):.6f}')
            print(f'  Target scalar mean: {val_metrics.get("target_scalar_mean", 0):.4f}')
            print(f'  Pred scalar mean: {val_metrics.get("pred_scalar_mean", 0):.4f}')

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            best_model_path = save_path / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'loss_type': loss_type,
                'weighting': weighting,
            }, best_model_path)
            print(f'  ✓ Best model saved (val_loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  Patience: {patience_counter}/{patience}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping after {epoch+1} epochs')
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f'  Checkpoint saved: {checkpoint_path.name}')

    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved to: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CTSM/CTSM-v time score estimator')
    parser.add_argument('--loss_type', type=str,
                       choices=['ctsm', 'ctsm_v'],
                       required=True,
                       help='Loss type: ctsm (scalar) or ctsm_v (vectorized)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256, 256-512 recommended for CTSM)')
    parser.add_argument('--lr', type=float, default=2e-3,
                       help='Learning rate (default: 2e-3)')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/ratio',
                       help='Save directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint (e.g., checkpoint_epoch_20.pt)')
    parser.add_argument('--weighting', type=str, default='time_score_norm',
                       choices=['time_score_norm', 'stein_score_norm', 'uniform'],
                       help='Weighting function (default: time_score_norm)')

    args = parser.parse_args()

    train_ctsm_estimator(
        loss_type=args.loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_timesteps=args.num_timesteps,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        patience=args.patience,
        resume_from=args.resume_from,
        weighting=args.weighting
    )
