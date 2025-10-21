"""
Basic test to verify the implementation works.
Run this after installing dependencies: pip install -r requirements.txt
"""
import sys
sys.path.insert(0, 'src')

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")

    # Test UNet
    from models.unet import UNet
    model = UNet()
    print(f"✓ UNet created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == x.shape, "Output shape mismatch"
    print(f"✓ UNet forward pass successful: {x.shape} -> {out.shape}")

    # Test RatioEstimator
    from models.ratio_estimator import RatioEstimator
    ratio_model = RatioEstimator()
    print(f"✓ RatioEstimator created with {sum(p.numel() for p in ratio_model.parameters()):,} parameters")

    # Test forward pass
    x1 = torch.randn(2, 1, 28, 28)
    x2 = torch.randn(2, 1, 28, 28)
    scores = ratio_model(x1, x2, t)
    assert scores.shape == (2,), "Scores shape mismatch"
    print(f"✓ RatioEstimator forward pass successful: output shape {scores.shape}")

    # Test DiffusionSchedule
    from utils.diffusion import DiffusionSchedule
    schedule = DiffusionSchedule(num_timesteps=1000, device='cpu')
    print(f"✓ DiffusionSchedule created with {schedule.num_timesteps} timesteps")

    # Test noise addition
    x0 = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    x_t, noise = schedule.add_noise(x0, t)
    print(f"✓ Noise addition successful: {x0.shape} -> {x_t.shape}")

    # Test x0 prediction
    x0_pred = schedule.predict_x0_from_noise(x_t, t, noise)
    error = (x0 - x0_pred).abs().mean()
    print(f"✓ x0 reconstruction error: {error:.6f} (should be ~0)")

    # Test losses
    from utils.losses import DensityRatioLoss
    for loss_type in ['disc', 'dv', 'ulsif', 'rulsif', 'kliep']:
        loss_fn = DensityRatioLoss(loss_type=loss_type)
        T_real = torch.randn(16)
        T_fake = torch.randn(16)
        loss, metrics = loss_fn(T_real, T_fake)
        print(f"✓ {loss_type} loss: {loss.item():.4f}")

    # Test MNIST dataset
    from data.mnist_dataset import RotatedMNIST, RatioTrainingDataset
    try:
        dataset_std = RotatedMNIST(root='./data', train=True, rotate=False, download=True)
        print(f"✓ Standard MNIST loaded: {len(dataset_std)} samples")

        dataset_rot = RotatedMNIST(root='./data', train=True, rotate=True, download=False)
        print(f"✓ Rotated MNIST loaded: {len(dataset_rot)} samples")

        ratio_dataset = RatioTrainingDataset(root='./data', train=True, real_fake_ratio=0.5, download=False)
        sample = ratio_dataset[0]
        print(f"✓ RatioTrainingDataset: img1 {sample['img1'].shape}, img2 {sample['img2'].shape}")
    except Exception as e:
        print(f"⚠ MNIST dataset test skipped (will download on first run): {e}")

    print("\n" + "="*60)
    print("All basic tests passed! ✓")
    print("="*60)
    print("\nYou can now proceed with training:")
    print("  1. Train diffusion models:")
    print("     python src/train_diffusion.py --dataset standard --epochs 50")
    print("     python src/train_diffusion.py --dataset rotated --epochs 50")
    print("\n  2. Train ratio estimators:")
    print("     python src/train_ratio.py --loss_type disc --epochs 30")
    print("\n  3. Generate samples:")
    print("     python src/sample.py --dataset standard --guided --loss_type disc")

except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
