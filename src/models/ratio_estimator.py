"""
Density-ratio (PMI) estimator for MNIST pairs.
Estimates log(p(x,y,t) / p(x,t)p(y,t)) for noisy MNIST pairs.
"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ImageEncoder(nn.Module):
    """
    Convolutional encoder for MNIST images (28x28).
    Maps image to a feature vector.
    """
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 7x7 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # 7x7 -> 3x3
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # Flatten
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Linear
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 28, 28]
        Returns:
            [B, feature_dim]
        """
        return self.encoder(x)


class RatioEstimator(nn.Module):
    """
    Density-ratio estimator for noisy MNIST image pairs.

    Takes two noisy images (e.g., standard and rotated MNIST) at the same
    timestep t and outputs a scalar score T(x, y, t).

    The score approximates:
    - For discriminator loss: logit ~ log(q/r)
    - For DV loss: T ~ log(q/r) + const
    - For uLSIF/RuLSIF/KLIEP: mapped to ratio w ~ q/r
    """
    def __init__(self, in_channels=1, feature_dim=256, hidden_dim=512, time_emb_dim=128):
        super().__init__()

        # Image encoders (shared architecture, separate parameters)
        self.encoder_1 = ImageEncoder(in_channels, feature_dim)
        self.encoder_2 = ImageEncoder(in_channels, feature_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # MLP to combine features and produce score
        self.score_network = nn.Sequential(
            nn.Linear(feature_dim * 2 + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x1, x2, t):
        """
        Args:
            x1: [B, 1, 28, 28] - first noisy image (e.g., standard MNIST)
            x2: [B, 1, 28, 28] - second noisy image (e.g., rotated MNIST)
            t: [B] - timesteps

        Returns:
            [B] - scalar scores T(x1, x2, t)
        """
        # Encode images
        features_1 = self.encoder_1(x1)  # [B, feature_dim]
        features_2 = self.encoder_2(x2)  # [B, feature_dim]

        # Time embedding
        t_emb = self.time_embed(t.float())  # [B, time_emb_dim]

        # Concatenate all features
        combined = torch.cat([features_1, features_2, t_emb], dim=-1)  # [B, 2*feature_dim + time_emb_dim]

        # Compute score
        score = self.score_network(combined).squeeze(-1)  # [B]

        return score


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = RatioEstimator()
    print(f"RatioEstimator parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    x1 = torch.randn(batch_size, 1, 28, 28)  # Standard MNIST
    x2 = torch.randn(batch_size, 1, 28, 28)  # Rotated MNIST
    t = torch.randint(0, 1000, (batch_size,))

    scores = model(x1, x2, t)
    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}, t={t.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Output scores: {scores}")
    assert scores.shape == (batch_size,), "Output should be [B] scalar scores"
    print("RatioEstimator test passed!")
