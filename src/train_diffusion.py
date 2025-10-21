"""
Train DDPM models on MNIST (standard or rotated).

Usage:
    python src/train_diffusion.py --dataset standard --epochs 50
    python src/train_diffusion.py --dataset rotated --epochs 50
"""
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from models.unet import UNet
from data.mnist_dataset import get_diffusion_dataloader
from utils.diffusion import DiffusionSchedule


def train_ddpm(
    dataset_type='standard',
    epochs=50,
    batch_size=128,
    lr=1e-4,
    num_timesteps=1000,
    device='cuda',
    save_dir='checkpoints/diffusion',
    num_workers=4
):
    """
    Train a DDPM model on MNIST.

    Args:
        dataset_type: 'standard' or 'rotated'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_timesteps: Number of diffusion timesteps
        device: Device to train on
        save_dir: Directory to save checkpoints
        num_workers: Number of data loading workers
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training DDPM on {dataset_type} MNIST")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")

    # Create save directory
    save_path = Path(save_dir) / dataset_type
    save_path.mkdir(parents=True, exist_ok=True)

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Diffusion schedule
    schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=device)

    # Data
    train_loader = get_diffusion_dataloader(
        dataset_type=dataset_type,
        batch_size=batch_size,
        train=True,
        num_workers=num_workers
    )
    val_loader = get_diffusion_dataloader(
        dataset_type=dataset_type,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, _ in pbar:
            images = images.to(device)
            batch_size_actual = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (batch_size_actual,), device=device)

            # Add noise
            noise = torch.randn_like(images)
            noisy_images, _ = schedule.add_noise(images, t, noise)

            # Predict noise
            noise_pred = model(noisy_images, t)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                batch_size_actual = images.shape[0]

                t = torch.randint(0, num_timesteps, (batch_size_actual,), device=device)
                noise = torch.randn_like(images)
                noisy_images, _ = schedule.add_noise(images, t, noise)

                noise_pred = model(noisy_images, t)
                loss = F.mse_loss(noise_pred, noise)

                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dataset_type': dataset_type,
                'num_timesteps': num_timesteps,
            }
            save_file = save_path / 'best_model.pt'
            torch.save(checkpoint, save_file)
            print(f'  -> Saved best model (val_loss: {best_val_loss:.4f})')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dataset_type': dataset_type,
                'num_timesteps': num_timesteps,
            }
            save_file = save_path / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, save_file)
            print(f'  -> Saved checkpoint at epoch {epoch+1}')

    print(f'\nTraining complete! Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved to: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPM on MNIST')
    parser.add_argument('--dataset', type=str, choices=['standard', 'rotated'], required=True,
                       help='Dataset type: standard or rotated MNIST')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/diffusion',
                       help='Directory to save checkpoints (default: checkpoints/diffusion)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    train_ddpm(
        dataset_type=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_timesteps=args.num_timesteps,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.num_workers
    )
