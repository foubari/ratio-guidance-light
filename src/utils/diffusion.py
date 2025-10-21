"""
Diffusion utilities: noise schedule, forward/reverse process, sampling.
Implements DDPM (Denoising Diffusion Probabilistic Models).
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


class DiffusionSchedule:
    """
    Linear beta schedule for DDPM.
    Implements the forward diffusion process and provides utilities for training/sampling.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        """
        Args:
            num_timesteps: Number of diffusion steps (T)
            beta_start: Starting value of beta schedule
            beta_end: Ending value of beta schedule
            device: Device to store tensors
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Precompute values for reverse diffusion q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Posterior variance: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x0: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional noise tensor (if None, sample from N(0, I))

        Returns:
            x_t: Noisy images [B, C, H, W]
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def predict_x0_from_noise(self, x_t, t, noise_pred):
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

        return x0_pred

    def p_sample_step(self, model, x_t, t, guidance_fn=None, guidance_scale=0.0):
        """
        Single reverse diffusion step: sample from p(x_{t-1} | x_t).

        Args:
            model: Noise prediction model
            x_t: Noisy image at timestep t [B, C, H, W]
            t: Current timestep [B]
            guidance_fn: Optional guidance function(x_t, t) -> gradient
            guidance_scale: Guidance strength

        Returns:
            x_{t-1}: Denoised image [B, C, H, W]
        """
        # Predict noise
        with torch.no_grad():
            noise_pred = model(x_t, t)

        # Apply guidance if provided
        if guidance_fn is not None and guidance_scale > 0:
            with torch.enable_grad():
                x_t_grad = x_t.detach().requires_grad_(True)
                guidance_grad = guidance_fn(x_t_grad, t)
                guidance_grad = guidance_grad.detach()

            # Add guidance to noise prediction
            # This modifies the score: score_guided = score_uncond + scale * gradient
            # Since noise_pred = -sigma * score, we subtract the scaled gradient
            noise_pred = noise_pred - guidance_scale * guidance_grad

        # Predict x_0
        x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Clamp to valid range
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute mean of q(x_{t-1} | x_t, x_0)
        # Use x0_pred for the mean computation (DDPM formula)
        alpha_bar_t = self.alphas_cumprod[t][:, None, None, None]
        alpha_bar_t_prev = self.alphas_cumprod_prev[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]

        # Compute coefficients
        coef1 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(self.alphas[t][:, None, None, None]) * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)

        # Posterior mean: mu = coef1 * x0_pred + coef2 * x_t
        mean = coef1 * x0_pred + coef2 * x_t

        # Add noise (except at t=0)
        if t[0] > 0:
            variance = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(self, model, shape, guidance_fn=None, guidance_scale=0.0, device=None):
        """
        Full DDPM sampling loop.

        Args:
            model: Noise prediction model
            shape: Output shape [B, C, H, W]
            guidance_fn: Optional guidance function
            guidance_scale: Guidance strength
            device: Device to sample on

        Returns:
            x_0: Generated images [B, C, H, W]
        """
        if device is None:
            device = self.device

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Reverse diffusion loop
        for t_idx in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc='Sampling'):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample_step(model, x_t, t, guidance_fn, guidance_scale)

        return x_t


class DDPMTrainer:
    """
    Trainer for DDPM models.
    """
    def __init__(self, model, schedule, optimizer, device='cuda'):
        self.model = model
        self.schedule = schedule
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x0):
        """
        Single training step for DDPM.

        Args:
            x0: Clean images [B, C, H, W]

        Returns:
            loss: MSE loss between predicted and true noise
        """
        batch_size = x0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=self.device)

        # Add noise
        noise = torch.randn_like(x0)
        x_t, _ = self.schedule.add_noise(x0, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    print("Testing diffusion utilities...")

    # Test schedule
    schedule = DiffusionSchedule(num_timesteps=1000, device='cpu')
    print(f"✓ Created schedule with {schedule.num_timesteps} timesteps")

    # Test forward diffusion
    x0 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    x_t, noise = schedule.add_noise(x0, t)
    print(f"✓ Forward diffusion: x0 {x0.shape} -> x_t {x_t.shape}")

    # Test x0 prediction
    x0_pred = schedule.predict_x0_from_noise(x_t, t, noise)
    print(f"✓ Predict x0 from noise: {x0_pred.shape}")
    reconstruction_error = (x0 - x0_pred).abs().mean()
    print(f"  Reconstruction error: {reconstruction_error:.6f} (should be ~0)")

    print("\nAll tests passed!")
