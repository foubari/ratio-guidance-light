# Ratio Guidance (Lightweight)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Project-orange" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
</p>

> **Status:** Research project - simplified implementation

## Overview

This is a lightweight, simplified version of the ratio_guidance project. It provides a streamlined implementation of density-ratio (PMI) guided diffusion using **MNIST** as a teaching example.

The key idea: Train two separate diffusion models (one on standard MNIST, one on 90-degree rotated MNIST), then use a learned density-ratio estimator to guide sampling from one model conditioned on samples from the other.

## Key Features

- Clean, educational codebase focused on core concepts
- MNIST-based (28x28 images) for fast experimentation
- 5 different density-ratio objectives: Discriminator, DV, uLSIF, RuLSIF, KLIEP
- Separate training scripts for diffusion models and ratio estimators
- Guided sampling with adjustable guidance scale

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ratio-guidance-light.git
cd ratio-guidance-light

# Create conda environment
conda create -n ratio-light python=3.9 -y
conda activate ratio-light

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Diffusion Models

First, train two DDPM models on standard and rotated MNIST:

```bash
# Train on standard MNIST
python src/train_diffusion.py --dataset standard --epochs 50

# Train on 90-degree rotated MNIST
python src/train_diffusion.py --dataset rotated --epochs 50
```

Models will be saved to `checkpoints/diffusion/`.

**During training:**
- Samples are generated every 10 epochs and saved to `checkpoints/diffusion/{dataset}/samples/`
- Checkpoints are saved every 10 epochs
- Best model is saved when validation loss improves

**After training, visualize samples:**
```bash
# Generate samples from both models
python src/visualize_diffusion_samples.py

# Generate from specific model
python src/visualize_diffusion_samples.py --dataset standard --num_samples 64
```

### 2. Train Ratio Estimators

Train density-ratio estimators with different loss functions:

```bash
# Discriminator loss (recommended for beginners)
python src/train_ratio.py --loss_type disc --epochs 30

# Or try other losses
python src/train_ratio.py --loss_type dv --epochs 30
python src/train_ratio.py --loss_type ulsif --epochs 30
python src/train_ratio.py --loss_type rulsif --epochs 30
python src/train_ratio.py --loss_type kliep --epochs 30
```

Models will be saved to `checkpoints/ratio/<loss_type>/`.

### 3. Generate Samples

```bash
# Unconditional sampling from standard MNIST
python src/sample.py --dataset standard --num_samples 16

# Guided sampling: generate standard MNIST conditioned on rotated MNIST
python src/sample.py --dataset standard --num_samples 16 \
    --guided --loss_type disc --guidance_scale 2.0 \
    --condition_dataset rotated
```

Outputs will be saved to `outputs/`.

## Project Structure

```
ratio-guidance-light/
├── data/                    # MNIST data (auto-downloaded)
├── checkpoints/            # Saved models
│   ├── diffusion/         # DDPM checkpoints
│   │   ├── standard/      # Standard MNIST model
│   │   └── rotated/       # Rotated MNIST model
│   └── ratio/             # Ratio estimator checkpoints
│       ├── disc/          # Discriminator loss
│       ├── dv/            # Donsker-Varadhan loss
│       ├── ulsif/         # uLSIF loss
│       ├── rulsif/        # RuLSIF loss
│       └── kliep/         # KLIEP loss
├── outputs/               # Generated samples
├── src/
│   ├── models/
│   │   ├── unet.py       # UNet for DDPM
│   │   └── ratio_estimator.py  # Density-ratio network
│   ├── data/
│   │   └── mnist_dataset.py    # MNIST loaders
│   ├── utils/
│   │   ├── diffusion.py       # Diffusion schedule & sampling
│   │   ├── losses.py          # Ratio losses
│   │   └── trainer.py         # Training utilities
│   ├── train_diffusion.py     # Train DDPM models
│   ├── train_ratio.py         # Train ratio estimators
│   ├── sample.py              # Guided sampling
│   └── visualize_diffusion_samples.py  # Visualize trained models
└── README.md
```

## Training Details

### Diffusion Models
- Architecture: Lightweight UNet (~5M parameters)
- Training: Standard DDPM objective (predict noise)
- Dataset: 28x28 grayscale MNIST
- Schedule: Linear beta schedule, 1000 timesteps

### Ratio Estimators
- Architecture: Conv encoder + MLP head (~1M parameters)
- Input: Two noisy images at the same timestep t
- Output: Scalar score T(x, y, t)
- Training data:
  - Real pairs: Same digit in standard and rotated form, noised to same t
  - Fake pairs: Random shuffling of standard/rotated samples

### Loss Functions
| Loss | Objective | Use Case |
|------|-----------|----------|
| **Discriminator** | Binary classification | Most stable, recommended for beginners |
| **DV** | Mutual information lower bound | Theoretical connections to MI |
| **uLSIF** | Least-squares density ratio | Direct ratio estimation |
| **RuLSIF** | Relative uLSIF | More stable with hyperparameter alpha |
| **KLIEP** | KL importance estimation | KL-based objective |

## Usage Examples

### Experiment with Guidance Scales

```bash
# Weak guidance
python src/sample.py --dataset standard --guided --guidance_scale 0.5

# Medium guidance
python src/sample.py --dataset standard --guided --guidance_scale 2.0

# Strong guidance
python src/sample.py --dataset standard --guided --guidance_scale 5.0
```

### Compare Different Loss Functions

```bash
for loss in disc dv ulsif rulsif kliep; do
    python src/sample.py --dataset standard --guided \
        --loss_type $loss --guidance_scale 2.0
done
```

## Key Differences from Full Version

- **Dataset**: MNIST (28x28) vs Night/Day images (64x64) or Audio/Image pairs
- **Modalities**: Single modality (rotated images) vs true multimodal (audio/image, night/day)
- **Architecture**: Smaller models (~5M params) vs full-scale models
- **Latent space**: Pixel-space only vs VAE latent space option
- **Focus**: Educational clarity vs production-ready research code

## Citation

If you use this code, please cite the original ratio_guidance project.

## License

MIT License

---

<p align="center">
  <em>A simplified research implementation for ratio-guided diffusion models.</em>
</p>
