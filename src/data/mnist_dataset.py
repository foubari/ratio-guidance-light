"""
MNIST dataset with rotation support for ratio guidance training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random


class RotatedMNIST(Dataset):
    """
    MNIST dataset with 90-degree rotation.
    Can return either standard or rotated version.
    """
    def __init__(self, root='./data', train=True, rotate=False, download=True):
        """
        Args:
            root: Data directory
            train: Whether to use train or test set
            rotate: If True, rotate all images 90 degrees clockwise
            download: Whether to download MNIST if not present
        """
        self.rotate = rotate

        # Standard MNIST transforms: resize to 28x28, convert to tensor, normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]
        ])

        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]

        if self.rotate:
            # Rotate 90 degrees clockwise
            # PyTorch uses (C, H, W) format, so we rotate on dims (1, 2)
            img = torch.rot90(img, k=-1, dims=(1, 2))

        return img, label


class RatioTrainingDataset(Dataset):
    """
    Dataset for training density-ratio estimator.
    Returns pairs of (standard, rotated) MNIST images.

    Real pairs: Same digit in standard and rotated form
    Fake pairs: Random shuffling to get product of marginals
    """
    def __init__(self, root='./data', train=True, real_fake_ratio=0.5, download=True):
        """
        Args:
            root: Data directory
            train: Whether to use train or test set
            real_fake_ratio: Probability of returning a real (corresponding) pair
            download: Whether to download MNIST if not present
        """
        self.real_fake_ratio = real_fake_ratio

        # Load both standard and rotated datasets
        self.dataset_standard = RotatedMNIST(root, train, rotate=False, download=download)
        self.dataset_rotated = RotatedMNIST(root, train, rotate=True, download=download)

        assert len(self.dataset_standard) == len(self.dataset_rotated)

    def __len__(self):
        return len(self.dataset_standard)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'img1': standard MNIST image [1, 28, 28]
                'img2': rotated MNIST image [1, 28, 28]
                'is_real': 1.0 if real pair (same digit), 0.0 if fake (shuffled)
                'label1': digit label for img1
                'label2': digit label for img2
        """
        # Decide if real or fake
        is_real = random.random() < self.real_fake_ratio

        # Get standard image
        img1, label1 = self.dataset_standard[idx]

        if is_real:
            # Real pair: use the same index for rotated
            img2, label2 = self.dataset_rotated[idx]
            assert label1 == label2, "Real pairs should have same label"
        else:
            # Fake pair: random index for rotated
            random_idx = random.randint(0, len(self.dataset_rotated) - 1)
            img2, label2 = self.dataset_rotated[random_idx]

        return {
            'img1': img1,
            'img2': img2,
            'is_real': torch.tensor(1.0 if is_real else 0.0),
            'label1': label1,
            'label2': label2
        }


def get_diffusion_dataloader(dataset_type='standard', batch_size=128, train=True, num_workers=4):
    """
    Get dataloader for training diffusion models.

    Args:
        dataset_type: 'standard' or 'rotated'
        batch_size: Batch size
        train: Whether to use train or test set
        num_workers: Number of data loading workers

    Returns:
        DataLoader
    """
    rotate = (dataset_type == 'rotated')

    dataset = RotatedMNIST(
        root='./data',
        train=train,
        rotate=rotate,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )

    return loader


def get_ratio_dataloader(batch_size=128, train=True, real_fake_ratio=0.5, num_workers=4):
    """
    Get dataloader for training ratio estimator.

    Args:
        batch_size: Batch size
        train: Whether to use train or test set
        real_fake_ratio: Probability of real pairs
        num_workers: Number of data loading workers

    Returns:
        DataLoader
    """
    dataset = RatioTrainingDataset(
        root='./data',
        train=train,
        real_fake_ratio=real_fake_ratio,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )

    return loader


if __name__ == "__main__":
    print("Testing MNIST datasets...")

    # Test standard MNIST
    print("\n1. Testing standard MNIST:")
    dataset_std = RotatedMNIST(rotate=False, train=True)
    img, label = dataset_std[0]
    print(f"  Shape: {img.shape}, Label: {label}, Range: [{img.min():.2f}, {img.max():.2f}]")

    # Test rotated MNIST
    print("\n2. Testing rotated MNIST:")
    dataset_rot = RotatedMNIST(rotate=True, train=True)
    img, label = dataset_rot[0]
    print(f"  Shape: {img.shape}, Label: {label}, Range: [{img.min():.2f}, {img.max():.2f}]")

    # Test ratio training dataset
    print("\n3. Testing ratio training dataset:")
    ratio_dataset = RatioTrainingDataset(train=True, real_fake_ratio=0.5)
    sample = ratio_dataset[0]
    print(f"  img1 shape: {sample['img1'].shape}")
    print(f"  img2 shape: {sample['img2'].shape}")
    print(f"  is_real: {sample['is_real'].item()}")
    print(f"  labels: {sample['label1']} vs {sample['label2']}")

    # Test dataloaders
    print("\n4. Testing diffusion dataloader:")
    loader = get_diffusion_dataloader('standard', batch_size=32, train=True, num_workers=0)
    batch = next(iter(loader))
    print(f"  Batch shape: {batch[0].shape}, Labels shape: {batch[1].shape}")

    print("\n5. Testing ratio dataloader:")
    loader = get_ratio_dataloader(batch_size=32, train=True, num_workers=0)
    batch = next(iter(loader))
    print(f"  img1: {batch['img1'].shape}")
    print(f"  img2: {batch['img2'].shape}")
    print(f"  is_real: {batch['is_real'].shape}")
    print(f"  Real pairs in batch: {batch['is_real'].sum().item()}/{len(batch['is_real'])}")

    print("\nAll tests passed!")
