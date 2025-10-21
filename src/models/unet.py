"""
Simplified UNet for MNIST diffusion (28x28 images).
Based on the architecture from the original ratio_guidance project but scaled down.
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


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None]

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + residual


class Downsample(nn.Module):
    """Downsample by 2x."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by 2x."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Simplified UNet for MNIST (28x28 grayscale images).

    Architecture:
    - 28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
    - Channels: 1 -> 64 -> 128 -> 256 -> 128 -> 64 -> 1
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        base_channels=64
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling path)
        # 28x28 -> 14x14 -> 7x7
        self.down1 = ResBlock(base_channels, base_channels, time_emb_dim)
        self.downsample1 = Downsample(base_channels)  # 28 -> 14

        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.downsample2 = Downsample(base_channels * 2)  # 14 -> 7

        # Bottleneck
        self.mid1 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid3 = ResBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        # Decoder (upsampling path)
        # 7x7 -> 14x14 -> 28x28
        self.upsample1 = Upsample(base_channels * 2)  # 7 -> 14
        self.up1 = ResBlock(base_channels * 4, base_channels, time_emb_dim)  # concat: 128*2

        self.upsample2 = Upsample(base_channels)  # 14 -> 28
        self.up2 = ResBlock(base_channels * 2, base_channels, time_emb_dim)  # concat: 64*2

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        """
        Args:
            x: [B, 1, 28, 28] - noisy images
            t: [B] - timesteps
        Returns:
            [B, 1, 28, 28] - predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial conv
        x = self.conv_in(x)  # [B, 64, 28, 28]

        # Encoder with skip connections
        skip1 = self.down1(x, t_emb)  # [B, 64, 28, 28]
        x = self.downsample1(skip1)   # [B, 64, 14, 14]

        skip2 = self.down2(x, t_emb)  # [B, 128, 14, 14]
        x = self.downsample2(skip2)   # [B, 128, 7, 7]

        # Bottleneck
        x = self.mid1(x, t_emb)  # [B, 256, 7, 7]
        x = self.mid2(x, t_emb)  # [B, 256, 7, 7]
        x = self.mid3(x, t_emb)  # [B, 128, 7, 7]

        # Decoder with skip connections
        x = self.upsample1(x)  # [B, 128, 14, 14]
        x = torch.cat([x, skip2], dim=1)  # [B, 256, 14, 14]
        x = self.up1(x, t_emb)  # [B, 64, 14, 14]

        x = self.upsample2(x)  # [B, 64, 28, 28]
        x = torch.cat([x, skip1], dim=1)  # [B, 128, 28, 28]
        x = self.up2(x, t_emb)  # [B, 64, 28, 28]

        # Output
        x = self.conv_out(x)  # [B, 1, 28, 28]

        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = UNet()
    print(f"UNet parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"
    print("UNet test passed!")
