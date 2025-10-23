"""
Train density-ratio estimator for MNIST pairs.

Usage:
    python src/train_ratio.py --loss_type disc --epochs 30
    python src/train_ratio.py --loss_type dv --epochs 30
    python src/train_ratio.py --loss_type ulsif --epochs 30
    python src/train_ratio.py --loss_type rulsif --epochs 30
    python src/train_ratio.py --loss_type kliep --epochs 30
    python src/train_ratio.py --loss_type infonce --epochs 30
"""
import argparse
import torch
from pathlib import Path

from models.ratio_estimator import RatioEstimator
from data.mnist_dataset import get_ratio_dataloader
from utils.diffusion import DiffusionSchedule
from utils.losses import DensityRatioLoss
from utils.trainer import RatioTrainer
from utils.path_utils import get_checkpoint_path


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
    patience=5,
    resume_from=None,
    # Loss-specific hyperparameters
    rulsif_alpha=0.2,
    rulsif_link='exp',
    kliep_lambda=1.0,
    infonce_tau=0.07,
    ulsif_l2=0.0,
    weight_decay=0.0
):
    """
    Train density-ratio estimator.

    Args:
        loss_type: One of ['disc', 'dv', 'ulsif', 'rulsif', 'kliep', 'infonce']
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        real_fake_ratio: Ratio of real to fake pairs in training data (InfoNCE uses only real pairs)
        num_timesteps: Number of diffusion timesteps
        device: Device to train on
        save_dir: Directory to save checkpoints
        num_workers: Number of data loading workers
        patience: Early stopping patience
        resume_from: Path to checkpoint to resume from (e.g., 'checkpoint_epoch_30.pt')
        rulsif_alpha: Alpha parameter for RuLSIF (default: 0.2)
        kliep_lambda: Lambda for KLIEP normalization penalty (default: 1.0)
        infonce_tau: Temperature for InfoNCE (default: 0.07)
        ulsif_l2: L2 regularization for uLSIF (default: 0.0, prefer weight_decay)
        weight_decay: Weight decay for optimizer (default: 0.0)
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Collect hyperparameters for path generation
    hyperparams = {
        'rulsif_alpha': rulsif_alpha,
        'rulsif_link': rulsif_link,
        'kliep_lambda': kliep_lambda,
        'infonce_tau': infonce_tau,
        'ulsif_l2': ulsif_l2,
        'use_exp_w': False,  # default, could be made configurable
    }

    # Create save directory with hyperparameter-based naming
    save_path = get_checkpoint_path(save_dir, loss_type, **hyperparams)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training ratio estimator with {loss_type} loss")
    print(f"Save path: {save_path}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Real/Fake ratio: {real_fake_ratio}")

    # Print non-default hyperparameters
    non_defaults = []
    if loss_type == "rulsif" and rulsif_alpha != 0.2:
        non_defaults.append(f"rulsif_alpha={rulsif_alpha}")
    if loss_type == "kliep" and kliep_lambda != 1.0:
        non_defaults.append(f"kliep_lambda={kliep_lambda}")
    if loss_type == "infonce" and infonce_tau != 0.07:
        non_defaults.append(f"infonce_tau={infonce_tau}")
    if loss_type == "ulsif" and ulsif_l2 > 0:
        non_defaults.append(f"ulsif_l2={ulsif_l2}")
    if non_defaults:
        print(f"Hyperparameters: {', '.join(non_defaults)}")

    # Model
    model = RatioEstimator().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function with hyperparameters
    loss_fn = DensityRatioLoss(
        loss_type=loss_type,
        rulsif_alpha=rulsif_alpha,
        rulsif_link=rulsif_link,
        kliep_lambda=kliep_lambda,
        infonce_tau=infonce_tau,
        ulsif_l2=ulsif_l2
    )

    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from is not None:
        checkpoint_path = save_path / resume_from
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_metrics', {}).get('loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch'] + 1}")
            print(f"Previous best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
    else:
        best_val_loss = float('inf')

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
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
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
            print(f'  E_q[w] (train): {train_metrics.get("E_q_w", train_metrics.get("E_q_w_alpha", 0)):.4f}')
            print(f'  E_r[w] (train): {train_metrics.get("E_r_w", train_metrics.get("E_r_w_alpha", 0)):.4f}')
        elif loss_type == 'kliep':
            print(f'  Constraint residual (train): {train_metrics.get("constraint_resid", 0):.4f}')
        elif loss_type == 'infonce':
            print(f'  Train Accuracy: {train_metrics.get("accuracy_avg", 0):.4f}')
            print(f'  Val Accuracy: {val_metrics.get("accuracy_avg", 0):.4f}')
            print(f'  Diag vs Off-diag (train): {train_metrics.get("diag_vs_offdiag", 0):.4f}')

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
                       choices=['disc', 'dv', 'ulsif', 'rulsif', 'kliep', 'infonce'],
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
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Checkpoint filename to resume from (e.g., checkpoint_epoch_30.pt)')

    # Loss-specific hyperparameters
    parser.add_argument('--rulsif_alpha', type=float, default=0.2,
                       help='Alpha parameter for RuLSIF (default: 0.2)')
    parser.add_argument('--rulsif_link', type=str, default='exp',
                       choices=['exp', 'softplus', 'identity'],
                       help='Link function for RuLSIF (default: exp)')
    parser.add_argument('--kliep_lambda', type=float, default=1.0,
                       help='Lambda parameter for KLIEP normalization penalty (default: 1.0, 0.0 for canonical)')
    parser.add_argument('--infonce_tau', type=float, default=0.07,
                       help='Temperature parameter for InfoNCE (default: 0.07)')
    parser.add_argument('--ulsif_l2', type=float, default=0.0,
                       help='L2 regularization for uLSIF (default: 0.0, use weight_decay instead)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for optimizer (default: 0.0)')

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
        patience=args.patience,
        resume_from=args.resume_from,
        rulsif_alpha=args.rulsif_alpha,
        rulsif_link=args.rulsif_link,
        kliep_lambda=args.kliep_lambda,
        infonce_tau=args.infonce_tau,
        ulsif_l2=args.ulsif_l2,
        weight_decay=args.weight_decay
    )
