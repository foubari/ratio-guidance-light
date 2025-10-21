"""
Train density-ratio estimator for MNIST pairs.

Usage:
    python src/train_ratio.py --loss_type disc --epochs 30
    python src/train_ratio.py --loss_type dv --epochs 30
    python src/train_ratio.py --loss_type ulsif --epochs 30
    python src/train_ratio.py --loss_type rulsif --epochs 30
    python src/train_ratio.py --loss_type kliep --epochs 30
"""
import argparse
import torch
from pathlib import Path

from models.ratio_estimator import RatioEstimator
from data.mnist_dataset import get_ratio_dataloader
from utils.diffusion import DiffusionSchedule
from utils.losses import DensityRatioLoss
from utils.trainer import RatioTrainer


def train_ratio_estimator(
    loss_type='disc',
    epochs=30,
    batch_size=128,
    lr=1e-4,
    real_fake_ratio=0.5,
    num_timesteps=1000,
    device='cuda',
    save_dir='checkpoints/ratio',
    num_workers=4,
    patience=5
):
    """
    Train density-ratio estimator.

    Args:
        loss_type: One of ['disc', 'dv', 'ulsif', 'rulsif', 'kliep']
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        real_fake_ratio: Ratio of real to fake pairs in training data
        num_timesteps: Number of diffusion timesteps
        device: Device to train on
        save_dir: Directory to save checkpoints
        num_workers: Number of data loading workers
        patience: Early stopping patience
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training ratio estimator with {loss_type} loss")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Real/Fake ratio: {real_fake_ratio}")

    # Create save directory
    save_path = Path(save_dir) / loss_type
    save_path.mkdir(parents=True, exist_ok=True)

    # Model
    model = RatioEstimator().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    loss_fn = DensityRatioLoss(loss_type=loss_type)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Diffusion schedule
    schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)

    # Data
    train_loader = get_ratio_dataloader(
        batch_size=batch_size,
        train=True,
        real_fake_ratio=real_fake_ratio,
        num_workers=num_workers
    )
    val_loader = get_ratio_dataloader(
        batch_size=batch_size,
        train=False,
        real_fake_ratio=real_fake_ratio,
        num_workers=num_workers
    )

    # Trainer
    trainer = RatioTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        diffusion_schedule=schedule,
        device=device
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, epochs)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Print metrics
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'  Train Loss: {train_metrics["loss"]:.4f}')
        print(f'  Val Loss: {val_metrics["loss"]:.4f}')

        # Print loss-specific metrics
        if loss_type == 'disc':
            print(f'  Train Acc: {train_metrics.get("total_acc", 0):.4f}')
            print(f'  Val Acc: {val_metrics.get("total_acc", 0):.4f}')
        elif loss_type == 'dv':
            print(f'  DV Bound (train): {train_metrics.get("dv_bound", 0):.4f}')
            print(f'  DV Bound (val): {val_metrics.get("dv_bound", 0):.4f}')
        elif loss_type in ['ulsif', 'rulsif']:
            print(f'  E_q[w] (train): {train_metrics.get("E_q_w", train_metrics.get("E_q_r_alpha", 0)):.4f}')
            print(f'  E_r[w] (train): {train_metrics.get("E_r_w", train_metrics.get("E_r_r_alpha", 0)):.4f}')
        elif loss_type == 'kliep':
            print(f'  Constraint residual (train): {train_metrics.get("constraint_resid", 0):.4f}')

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'loss_type': loss_type,
                'num_timesteps': num_timesteps,
            }
            save_file = save_path / 'best_model.pt'
            torch.save(checkpoint, save_file)
            print(f'  -> Saved best model (val_loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  -> No improvement. Patience: {patience_counter}/{patience}')

            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'loss_type': loss_type,
                'num_timesteps': num_timesteps,
            }
            save_file = save_path / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, save_file)
            print(f'  -> Saved checkpoint at epoch {epoch+1}')

    print(f'\nTraining complete! Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved to: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train density-ratio estimator')
    parser.add_argument('--loss_type', type=str,
                       choices=['disc', 'dv', 'ulsif', 'rulsif', 'kliep'],
                       required=True,
                       help='Loss type for density-ratio estimation')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--real_fake_ratio', type=float, default=0.5,
                       help='Ratio of real pairs in training data (default: 0.5)')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/ratio',
                       help='Directory to save checkpoints (default: checkpoints/ratio)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')

    args = parser.parse_args()

    train_ratio_estimator(
        loss_type=args.loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        real_fake_ratio=args.real_fake_ratio,
        num_timesteps=args.num_timesteps,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        patience=args.patience
    )
