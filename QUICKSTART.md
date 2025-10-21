# Quick Start Guide

This guide will help you get started with ratio-guidance-light.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, but CPU works too)

## Installation

```bash
# Clone the repository
git clone https://github.com/foubari/ratio-guidance-light.git
cd ratio-guidance-light

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

Run the basic test to ensure everything is set up correctly:

```bash
python test_basic.py
```

This will test all core components without training.

## Training Workflow

### Step 1: Train Diffusion Models (Required)

Train two DDPM models - one on standard MNIST, one on 90-degree rotated MNIST:

```bash
# Train on standard MNIST (~10-15 min on GPU, 50 epochs)
python src/train_diffusion.py --dataset standard --epochs 50 --batch_size 128

# Train on rotated MNIST (~10-15 min on GPU, 50 epochs)
python src/train_diffusion.py --dataset rotated --epochs 50 --batch_size 128
```

**Models saved to:** `checkpoints/diffusion/{standard,rotated}/best_model.pt`

### Step 2: Train Ratio Estimators (Required for Guidance)

Train density-ratio estimators to enable guided sampling:

```bash
# Start with discriminator loss (most stable)
python src/train_ratio.py --loss_type disc --epochs 30 --batch_size 128

# Optional: Try other losses
python src/train_ratio.py --loss_type dv --epochs 30
python src/train_ratio.py --loss_type ulsif --epochs 30
python src/train_ratio.py --loss_type rulsif --epochs 30
python src/train_ratio.py --loss_type kliep --epochs 30
```

**Models saved to:** `checkpoints/ratio/{loss_type}/best_model.pt`

### Step 3: Generate Samples

#### Unconditional Sampling

Generate samples without guidance:

```bash
python src/sample.py --dataset standard --num_samples 16
```

#### Guided Sampling

Generate samples with ratio-based guidance:

```bash
# Basic guided sampling
python src/sample.py --dataset standard --num_samples 16 \
    --guided --loss_type disc --guidance_scale 2.0 \
    --condition_dataset rotated

# Experiment with guidance strength
python src/sample.py --dataset standard --guided \
    --guidance_scale 0.5  # Weak guidance

python src/sample.py --dataset standard --guided \
    --guidance_scale 5.0  # Strong guidance
```

**Outputs saved to:** `outputs/`

## Quick Debug Run

To quickly verify everything works (2 epochs, small batch):

```bash
# Quick diffusion training test (~2 min)
python src/train_diffusion.py --dataset standard --epochs 2 --batch_size 32

# Quick ratio training test (~1 min)
python src/train_ratio.py --loss_type disc --epochs 2 --batch_size 32
```

## Understanding the Output

### Training Output
- **Train Loss**: MSE between predicted and true noise (should decrease)
- **Val Loss**: Validation loss (should decrease)
- **Checkpoints**: Saved at `best_model.pt` and every 10 epochs

### Ratio Training Output
- **Discriminator**: Should see accuracy increasing (target: >0.8)
- **DV**: Should see DV bound increasing (higher = better)
- **uLSIF/RuLSIF/KLIEP**: Should see loss decreasing

### Generated Samples
- **Unconditional**: Random MNIST digits
- **Guided**: Shows condition image (rotated) and generated image (standard) pairs
  - With good guidance, generated digits should match the rotated condition

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python src/train_diffusion.py --dataset standard --batch_size 64
```

### CPU Training
The code automatically uses CPU if CUDA is not available. Training will be slower (~10x) but works fine.

### MNIST Download Issues
If MNIST download fails, manually download from:
http://yann.lecun.com/exdb/mnist/

Extract to `./data/MNIST/raw/`

## Expected Results

After training:
- **Diffusion models**: Should generate recognizable MNIST digits
- **Guided sampling**: Generated digits should correspond to rotated conditions
- **Guidance scale**: Higher values = stronger adherence to condition

## Next Steps

1. **Experiment with guidance scales**: Try 0.5, 1.0, 2.0, 5.0
2. **Compare loss functions**: Train all 5 ratio estimators and compare
3. **Visualize results**: Check `outputs/` directory
4. **Modify architecture**: Edit `src/models/` files
5. **Try different schedules**: Modify `DiffusionSchedule` parameters

## Common Commands Reference

```bash
# Training
python src/train_diffusion.py --dataset [standard|rotated] --epochs 50
python src/train_ratio.py --loss_type [disc|dv|ulsif|rulsif|kliep] --epochs 30

# Sampling
python src/sample.py --dataset [standard|rotated] --num_samples 16
python src/sample.py --dataset standard --guided --loss_type disc --guidance_scale 2.0

# Help
python src/train_diffusion.py --help
python src/train_ratio.py --help
python src/sample.py --help
```

## Performance Benchmarks

On NVIDIA RTX 3090:
- Diffusion training: ~10 min (50 epochs)
- Ratio training: ~5 min (30 epochs)
- Sampling: ~30 sec (16 samples, 1000 steps)

On CPU (modern):
- Diffusion training: ~2 hours (50 epochs)
- Ratio training: ~30 min (30 epochs)
- Sampling: ~5 min (16 samples, 1000 steps)
