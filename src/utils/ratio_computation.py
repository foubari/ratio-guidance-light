"""
Utilities for computing log density ratio from time score estimates.

Based on: "Density Ratio Estimation with Conditional Probability Paths" (Yu et al., 2025)

Key formula: log(p₁(x)/p₀(x)) = ∫₀¹ ∂_t log p_t(x) dt
"""
import torch


def compute_log_ratio_from_time_score(model, x, y, num_steps=100, device='cuda'):
    """
    Compute log(p(x,y)/p(x)p(y)) by integrating the time score.

    Uses: log ratio = ∫₀¹ ∂_t log p_t(x,y) dt

    Args:
        model: TimeScoreEstimator (trained CTSM model)
        x: image from standard MNIST (B, 1, 28, 28)
        y: image from rotated MNIST (B, 1, 28, 28)
        num_steps: number of integration steps (50-100 recommended)
        device: device

    Returns:
        log_ratio: (B,) log density ratio
    """
    model.eval()

    with torch.no_grad():
        # Integration using trapezoidal rule
        t_values = torch.linspace(0, 1, num_steps, device=device)
        time_scores = []

        for t_val in t_values:
            t_batch = t_val.repeat(x.shape[0])
            time_score = model(x, y, t_batch)  # (B,)
            time_scores.append(time_score)

        time_scores = torch.stack(time_scores, dim=1)  # (B, num_steps)

        # Trapezoidal integration: ∫ f(t) dt ≈ Σ (f(t_i) + f(t_{i+1})) * dt / 2
        dt = 1.0 / (num_steps - 1)
        log_ratio = torch.trapz(time_scores, dx=dt, dim=1)

    return log_ratio


def compute_log_ratio_from_vectorized_time_score(model, x, y, num_steps=100, device='cuda'):
    """
    Compute log(p(x,y)/p(x)p(y)) from CTSM-v by summing then integrating.

    For vectorized time score: ∂_t log p_t = Σ_i vec(∂_t log p_t)_i

    Args:
        model: VectorizedTimeScoreEstimator (trained CTSM-v model)
        x: image (B, 1, 28, 28)
        y: image (B, 1, 28, 28)
        num_steps: number of integration steps
        device: device

    Returns:
        log_ratio: (B,) log density ratio
    """
    model.eval()

    with torch.no_grad():
        t_values = torch.linspace(0, 1, num_steps, device=device)
        time_scores = []

        for t_val in t_values:
            t_batch = t_val.repeat(x.shape[0])
            vec_time_score = model(x, y, t_batch)  # (B, D)
            # Sum over dimensions to get scalar time score
            time_score = vec_time_score.sum(dim=-1)  # (B,)
            time_scores.append(time_score)

        time_scores = torch.stack(time_scores, dim=1)  # (B, num_steps)

        # Integrate
        dt = 1.0 / (num_steps - 1)
        log_ratio = torch.trapz(time_scores, dx=dt, dim=1)

    return log_ratio


def compute_gradient_of_log_ratio(model, x, y, num_steps=100, device='cuda'):
    """
    Compute ∇_x log ratio and ∇_y log ratio via autodiff.

    Method: Differentiate the integrated time score.

    Args:
        model: TimeScoreEstimator or VectorizedTimeScoreEstimator
        x: image (B, 1, 28, 28) - requires_grad=True
        y: image (B, 1, 28, 28) - requires_grad=True
        num_steps: integration steps
        device: device

    Returns:
        grad_x: (B, 1, 28, 28) - ∇_x log ratio
        grad_y: (B, 1, 28, 28) - ∇_y log ratio
    """
    # Ensure gradients are enabled
    x.requires_grad_(True)
    y.requires_grad_(True)

    # Determine model type
    from models.ratio_estimator import VectorizedTimeScoreEstimator

    # Compute log ratio
    if isinstance(model, VectorizedTimeScoreEstimator):
        log_ratio = compute_log_ratio_from_vectorized_time_score(model, x, y, num_steps, device)
    else:
        log_ratio = compute_log_ratio_from_time_score(model, x, y, num_steps, device)

    # Compute gradients
    grad_x = torch.autograd.grad(log_ratio.sum(), x, create_graph=True)[0]
    grad_y = torch.autograd.grad(log_ratio.sum(), y, create_graph=True)[0]

    return grad_x, grad_y


if __name__ == "__main__":
    print("="*60)
    print("Testing ratio computation utilities")
    print("="*60)

    # Note: These are unit tests with dummy models
    # Real tests would use trained models

    from models.ratio_estimator import TimeScoreEstimator, VectorizedTimeScoreEstimator

    batch_size = 4
    device = 'cpu'

    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randn(batch_size, 1, 28, 28, device=device)

    print("\n" + "-"*60)
    print("Test 1: Log ratio from TimeScoreEstimator")
    print("-"*60)

    model_ctsm = TimeScoreEstimator().to(device)
    log_ratio = compute_log_ratio_from_time_score(model_ctsm, x, y, num_steps=50, device=device)
    print(f"Log ratio shape: {log_ratio.shape}")
    print(f"Log ratio range: [{log_ratio.min():.3f}, {log_ratio.max():.3f}]")
    assert log_ratio.shape == (batch_size,)
    print("✓ Time score integration OK")

    print("\n" + "-"*60)
    print("Test 2: Log ratio from VectorizedTimeScoreEstimator")
    print("-"*60)

    model_ctsm_v = VectorizedTimeScoreEstimator().to(device)
    log_ratio_v = compute_log_ratio_from_vectorized_time_score(model_ctsm_v, x, y, num_steps=50, device=device)
    print(f"Log ratio shape: {log_ratio_v.shape}")
    print(f"Log ratio range: [{log_ratio_v.min():.3f}, {log_ratio_v.max():.3f}]")
    assert log_ratio_v.shape == (batch_size,)
    print("✓ Vectorized time score integration OK")

    print("\n" + "-"*60)
    print("Test 3: Gradient computation")
    print("-"*60)

    x_grad = x.clone().requires_grad_(True)
    y_grad = y.clone().requires_grad_(True)

    grad_x, grad_y = compute_gradient_of_log_ratio(model_ctsm, x_grad, y_grad, num_steps=50, device=device)

    print(f"grad_x shape: {grad_x.shape}")
    print(f"grad_y shape: {grad_y.shape}")
    print(f"grad_x has NaN: {torch.isnan(grad_x).any().item()}")
    print(f"grad_y has NaN: {torch.isnan(grad_y).any().item()}")
    print(f"grad_x norm: {grad_x.norm():.4f}")
    print(f"grad_y norm: {grad_y.norm():.4f}")

    assert not torch.isnan(grad_x).any(), "grad_x should not have NaN"
    assert not torch.isnan(grad_y).any(), "grad_y should not have NaN"
    print("✓ Gradient computation OK")

    print("\n" + "="*60)
    print("All ratio computation tests passed!")
    print("="*60)
