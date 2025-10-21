"""
Simple CNN classifier for MNIST digit recognition.
Used for evaluating guided sampling accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    Simple CNN classifier for MNIST (28x28 grayscale).

    Architecture:
    - 2 conv layers with max pooling
    - 2 fully connected layers
    - ~100k parameters
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 2 pooling: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 28, 28] - grayscale images in range [-1, 1]
        Returns:
            logits: [B, 10] - class logits
        """
        # Normalize from [-1, 1] to [0, 1] for consistency with standard MNIST
        x = (x + 1.0) / 2.0

        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: [B, 64, 7, 7] -> [B, 64*7*7]
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        """
        Get predicted class labels.

        Args:
            x: [B, 1, 28, 28]
        Returns:
            predictions: [B] - predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the classifier
    model = MNISTClassifier()
    print(f"Classifier parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 1, 28, 28) * 2 - 1  # Range [-1, 1]
    logits = model(x)
    predictions = model.predict(x)

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {predictions}")
    assert logits.shape == (4, 10)
    assert predictions.shape == (4,)
    print("Classifier test passed!")
