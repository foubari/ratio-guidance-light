"""
Train MNIST classifiers for evaluation.

Usage:
    # Train classifier for standard MNIST
    python src/train_classifier.py --dataset standard

    # Train classifier for rotated MNIST
    python src/train_classifier.py --dataset rotated

    # Train both (recommended)
    python src/train_classifier.py --dataset both
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from models.classifier import MNISTClassifier
from data.mnist_dataset import get_diffusion_dataloader


def train_classifier(
    dataset_type='standard',
    epochs=10,
    batch_size=128,
    lr=1e-3,
    device='cuda',
    save_dir='checkpoints/classifiers',
    num_workers=4
):
    """
    Train a classifier on MNIST.

    Args:
        dataset_type: 'standard', 'rotated', or 'both'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        num_workers: Number of data loading workers
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training classifier on {dataset_type} MNIST")
    print(f"Device: {device}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to train on
    datasets_to_train = ['standard', 'rotated'] if dataset_type == 'both' else [dataset_type]

    for ds_type in datasets_to_train:
        print(f"\n{'='*60}")
        print(f"Training classifier for {ds_type} MNIST")
        print(f"{'='*60}")

        # Model
        model = MNISTClassifier(num_classes=10).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Data
        train_loader = get_diffusion_dataloader(
            dataset_type=ds_type,
            batch_size=batch_size,
            train=True,
            num_workers=num_workers
        )
        val_loader = get_diffusion_dataloader(
            dataset_type=ds_type,
            batch_size=batch_size,
            train=False,
            num_workers=num_workers
        )

        # Training loop
        best_val_acc = 0.0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                logits = model(images)
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * train_correct / train_total:.2f}%'
                })

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': avg_val_loss,
                    'dataset_type': ds_type,
                }
                save_file = save_path / f'{ds_type}_classifier.pt'
                torch.save(checkpoint, save_file)
                print(f'  -> Saved best model (val_acc: {best_val_acc:.2f}%)')

        print(f'\nTraining complete for {ds_type}!')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        print(f'Model saved to: {save_path / f"{ds_type}_classifier.pt"}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--dataset', type=str,
                       choices=['standard', 'rotated', 'both'],
                       default='both',
                       help='Which dataset to train on (default: both)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/classifiers',
                       help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    train_classifier(
        dataset_type=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.num_workers
    )
