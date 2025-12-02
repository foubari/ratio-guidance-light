"""
CTSM (Conditional Time Score Matching) losses and utilities.

Based on: "Density Ratio Estimation with Conditional Probability Paths" (Yu et al., 2025)
ArXiv: https://arxiv.org/abs/2502.02300v3

Key innovation: Estimates log density ratio via time score integration:
    log(p₁(x)/p₀(x)) = ∫₀¹ ∂_t log p_t(x) dt

Advantages over TSM:
- 3x faster (no double autodiff)
- Better accuracy on high-dim tasks
- More stable training
- Closed-form target for regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_time_score_normalization(t, c=1.0, alpha_schedule='linear'):
    """
    Compute λ(t) for time score normalization weighting (Eq. 21 of paper).

    For VP path with linear schedule (α_t = t):
        λ(t) = (1 - t²)² / (2t²·(α'_t)² + (α'_t)²·(1-t²)·c)

    Args:
        t: time in [0, 1] (B,)
        c: constant (default=1.0 works well empirically)
        alpha_schedule: 'linear' (α_t = t, α'_t = 1)

    Returns:
        lambda_t: weighting (B,)
    """
    if alpha_schedule == 'linear':
        # α_t = t, α'_t = 1
        alpha_t = t
        alpha_prime_t = torch.ones_like(t)
    else:
        raise NotImplementedError(f"Schedule {alpha_schedule} not implemented")

    # Compute weighting
    numerator = (1 - alpha_t**2) ** 2
    denominator = 2 * alpha_t**2 * alpha_prime_t**2 + alpha_prime_t**2 * (1 - alpha_t**2) * c

    # Add epsilon for numerical stability
    lambda_t = numerator / (denominator + 1e-8)

    return lambda_t


def compute_conditional_time_score_vp(x_noisy, x_clean, t, alpha_schedule='linear'):
    """
    Compute ∂_t log p_t(x|z) in closed form for VP path (Eq. 62 of paper).

    This is the TARGET for CTSM training - no learning involved!

    Formula for linear VP path:
        ∂_t log p_t(x|z) = D·(α_t·α'_t)/(1-α_t²)
                           - (α_t·α'_t)/(1-α_t²)·||ε||²
                           + (1/√(1-α_t²))·ε^T·(α'_t·z)

    where ε = (x - α_t·z) / √(1-α_t²)

    Args:
        x_noisy: x_t (noisy images) (B, C, H, W)
        x_clean: x_1 (clean images, z) (B, C, H, W)
        t: time in [0, 1] (B,)
        alpha_schedule: 'linear' (α_t = t)

    Returns:
        time_score: scalar per sample (B,)
    """
    B = x_noisy.shape[0]
    D = x_noisy.numel() // B  # Total number of pixels

    # Flatten to (B, D)
    x_noisy_flat = x_noisy.view(B, -1)
    x_clean_flat = x_clean.view(B, -1)

    # Compute α_t and α'_t
    if alpha_schedule == 'linear':
        alpha_t = t  # α_t = t
        alpha_prime_t = torch.ones_like(t)  # α'_t = 1
    else:
        raise NotImplementedError(f"Schedule {alpha_schedule} not implemented")

    # Expand to (B, 1)
    alpha_t = alpha_t.unsqueeze(-1)
    alpha_prime_t = alpha_prime_t.unsqueeze(-1)

    # Compute noise: ε = (x - α_t·x_1) / √(1-α_t²)
    epsilon = (x_noisy_flat - alpha_t * x_clean_flat) / torch.sqrt(1 - alpha_t**2 + 1e-8)

    # Three terms of Eq. 62
    term1 = D * (alpha_t * alpha_prime_t) / (1 - alpha_t**2 + 1e-8)
    term2 = -(alpha_t * alpha_prime_t) / (1 - alpha_t**2 + 1e-8) * (epsilon ** 2).sum(dim=-1, keepdim=True)
    term3 = (1 / torch.sqrt(1 - alpha_t**2 + 1e-8)) * (epsilon * (alpha_prime_t * x_clean_flat)).sum(dim=-1, keepdim=True)

    time_score = term1 + term2 + term3  # (B, 1)

    return time_score.squeeze(-1)  # (B,)


def compute_vectorized_conditional_time_score_vp(x_noisy, x_clean, t, alpha_schedule='linear'):
    """
    Compute vec(∂_t log p_t(x|z)) in closed form for VP path (Eq. 64 of paper).

    This is the TARGET for CTSM-v training - returns a VECTOR per sample.

    Formula (vectorized version of Eq. 62):
        vec(∂_t log p_t(x|z))_i = (α_t·α'_t)/(1-α_t²)
                                  - (α_t·α'_t)/(1-α_t²)·ε_i²
                                  + (1/√(1-α_t²))·ε_i·(α'_t·z_i)

    Args:
        x_noisy: x_t (noisy images) (B, C, H, W)
        x_clean: x_1 (clean images, z) (B, C, H, W)
        t: time in [0, 1] (B,)
        alpha_schedule: 'linear'

    Returns:
        vec_time_score: vector per sample (B, D) where D = C*H*W
    """
    B = x_noisy.shape[0]

    # Flatten to (B, D)
    x_noisy_flat = x_noisy.view(B, -1)
    x_clean_flat = x_clean.view(B, -1)

    # Compute α_t and α'_t
    if alpha_schedule == 'linear':
        alpha_t = t
        alpha_prime_t = torch.ones_like(t)
    else:
        raise NotImplementedError(f"Schedule {alpha_schedule} not implemented")

    # Expand to (B, 1)
    alpha_t = alpha_t.unsqueeze(-1)
    alpha_prime_t = alpha_prime_t.unsqueeze(-1)

    # Compute noise (element-wise)
    epsilon = (x_noisy_flat - alpha_t * x_clean_flat) / torch.sqrt(1 - alpha_t**2 + 1e-8)

    # Three terms - now vectorized (Eq. 64)
    term1 = (alpha_t * alpha_prime_t) / (1 - alpha_t**2 + 1e-8)  # (B, 1) - broadcasted
    term2 = -(alpha_t * alpha_prime_t) / (1 - alpha_t**2 + 1e-8) * (epsilon ** 2)  # (B, D)
    term3 = (1 / torch.sqrt(1 - alpha_t**2 + 1e-8)) * epsilon * (alpha_prime_t * x_clean_flat)  # (B, D)

    vec_time_score = term1 + term2 + term3  # (B, D)

    return vec_time_score


def ctsm_loss(model_output, target_time_score, t, weighting='time_score_norm', alpha_schedule='linear'):
    """
    CTSM (Conditional Time Score Matching) loss.

    Formula: L = E[λ(t) ||∂_t log p_t(x|z) - s_θ(x,t)||²]

    Args:
        model_output: predicted time score s_θ(x,t) (B,)
        target_time_score: ∂_t log p_t(x|z) computed in closed form (B,)
        t: time (B,)
        weighting: 'time_score_norm' (recommended), 'stein_score_norm', or 'uniform'
        alpha_schedule: 'linear'

    Returns:
        loss: scalar
        metrics: dict with detailed info
    """
    # Compute weighting function λ(t)
    if weighting == 'time_score_norm':
        lambda_t = compute_time_score_normalization(t, c=1.0, alpha_schedule=alpha_schedule)
    elif weighting == 'stein_score_norm':
        # For VP path: λ(t) = 1 - t²
        lambda_t = 1 - t**2
    elif weighting == 'uniform':
        lambda_t = torch.ones_like(t)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # CTSM loss: E[λ(t) ||target - prediction||²]
    squared_error = (target_time_score - model_output) ** 2
    loss = (lambda_t * squared_error).mean()

    # Metrics
    metrics = {
        "ctsm_loss": loss.detach().item(),
        "target_time_score_mean": target_time_score.mean().item(),
        "pred_time_score_mean": model_output.mean().item(),
        "target_time_score_std": target_time_score.std().item() if target_time_score.numel() > 1 else 0,
        "pred_time_score_std": model_output.std().item() if model_output.numel() > 1 else 0,
        "mse": squared_error.mean().item(),
        "lambda_t_mean": lambda_t.mean().item(),
    }

    return loss, metrics


def ctsm_v_loss(model_output, target_vec_time_score, t, weighting='time_score_norm', alpha_schedule='linear'):
    """
    CTSM-v (Vectorized Conditional Time Score Matching) loss.

    Formula: L = E[λ(t) ||vec(∂_t log p_t(x|z)) - s^vec_θ(x,t)||²]

    Args:
        model_output: predicted vectorized time score s^vec_θ(x,t) (B, D)
        target_vec_time_score: vec(∂_t log p_t(x|z)) (B, D)
        t: time (B,)
        weighting: 'time_score_norm' (recommended), 'stein_score_norm', or 'uniform'
        alpha_schedule: 'linear'

    Returns:
        loss: scalar
        metrics: dict
    """
    # Compute weighting function λ(t)
    if weighting == 'time_score_norm':
        lambda_t = compute_time_score_normalization(t, c=1.0, alpha_schedule=alpha_schedule)
    elif weighting == 'stein_score_norm':
        lambda_t = 1 - t**2
    elif weighting == 'uniform':
        lambda_t = torch.ones_like(t)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Expand lambda_t to match (B, D)
    lambda_t = lambda_t.unsqueeze(-1)  # (B, 1)

    # CTSM-v loss: E[λ(t) ||target - prediction||²]
    squared_error = (target_vec_time_score - model_output) ** 2
    weighted_squared_error = lambda_t * squared_error

    # Sum over dimensions, then average over batch
    loss = weighted_squared_error.sum(dim=-1).mean()

    # For comparison: sum vectorized scores to get scalar approximation
    target_scalar = target_vec_time_score.sum(dim=-1)
    pred_scalar = model_output.sum(dim=-1)

    # Metrics
    metrics = {
        "ctsm_v_loss": loss.detach().item(),
        "target_vec_norm": target_vec_time_score.norm(dim=-1).mean().item(),
        "pred_vec_norm": model_output.norm(dim=-1).mean().item(),
        "target_scalar_mean": target_scalar.mean().item(),
        "pred_scalar_mean": pred_scalar.mean().item(),
        "mse_per_dim": squared_error.mean().item(),
        "lambda_t_mean": lambda_t.mean().item(),
    }

    return loss, metrics


if __name__ == "__main__":
    print("="*60)
    print("Testing CTSM loss functions")
    print("="*60)

    # Test parameters
    batch_size = 8
    img_channels = 1
    img_size = 28

    # Create synthetic data
    x_noisy = torch.randn(batch_size, img_channels, img_size, img_size)
    x_clean = torch.randn(batch_size, img_channels, img_size, img_size)
    t = torch.rand(batch_size)  # Uniform in [0, 1]

    print("\n" + "-"*60)
    print("Test 1: Closed-form time score computation")
    print("-"*60)

    # Compute target time score (scalar)
    time_score = compute_conditional_time_score_vp(x_noisy, x_clean, t)
    print(f"Time score shape: {time_score.shape}")
    print(f"Time score range: [{time_score.min():.2f}, {time_score.max():.2f}]")
    print(f"Time score mean: {time_score.mean():.2f}")
    assert time_score.shape == (batch_size,), "Should be (B,)"
    print("✓ Scalar time score computation OK")

    # Compute vectorized time score
    vec_time_score = compute_vectorized_conditional_time_score_vp(x_noisy, x_clean, t)
    print(f"\nVectorized time score shape: {vec_time_score.shape}")
    print(f"Vec time score range: [{vec_time_score.min():.2f}, {vec_time_score.max():.2f}]")
    print(f"Sum of vec (should ≈ scalar): {vec_time_score.sum(dim=-1)}")
    print(f"Scalar time score: {time_score}")
    assert vec_time_score.shape == (batch_size, img_size * img_size), "Should be (B, 784)"
    print("✓ Vectorized time score computation OK")

    print("\n" + "-"*60)
    print("Test 2: Time score normalization weighting")
    print("-"*60)

    lambda_t = compute_time_score_normalization(t, c=1.0)
    print(f"Lambda(t) shape: {lambda_t.shape}")
    print(f"Lambda(t) range: [{lambda_t.min():.4f}, {lambda_t.max():.4f}]")
    print(f"Lambda(t) mean: {lambda_t.mean():.4f}")
    print("✓ Weighting function OK")

    print("\n" + "-"*60)
    print("Test 3: CTSM loss")
    print("-"*60)

    # Simulate model predictions
    pred_time_score = torch.randn_like(time_score)

    loss, metrics = ctsm_loss(pred_time_score, time_score, t, weighting='time_score_norm')
    print(f"CTSM loss: {loss.item():.4f}")
    print(f"Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("✓ CTSM loss computation OK")

    print("\n" + "-"*60)
    print("Test 4: CTSM-v loss")
    print("-"*60)

    # Simulate model predictions
    pred_vec_time_score = torch.randn_like(vec_time_score)

    loss_v, metrics_v = ctsm_v_loss(pred_vec_time_score, vec_time_score, t, weighting='time_score_norm')
    print(f"CTSM-v loss: {loss_v.item():.4f}")
    print(f"Metrics:")
    for k, v in metrics_v.items():
        print(f"  {k}: {v:.4f}")
    print("✓ CTSM-v loss computation OK")

    print("\n" + "-"*60)
    print("Test 5: Gradient flow")
    print("-"*60)

    # Test that gradients flow correctly
    pred_time_score_grad = torch.randn_like(time_score, requires_grad=True)
    loss, _ = ctsm_loss(pred_time_score_grad, time_score, t)
    loss.backward()

    print(f"Gradient shape: {pred_time_score_grad.grad.shape}")
    print(f"Gradient has NaN: {torch.isnan(pred_time_score_grad.grad).any().item()}")
    print(f"Gradient norm: {pred_time_score_grad.grad.norm().item():.4f}")
    assert not torch.isnan(pred_time_score_grad.grad).any(), "Gradients should not be NaN"
    print("✓ Gradient flow OK")

    print("\n" + "="*60)
    print("All CTSM tests passed!")
    print("="*60)
