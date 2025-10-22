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
- **6 different density-ratio objectives:** Discriminator, DV, uLSIF, RuLSIF, KLIEP, InfoNCE
- Separate training scripts for diffusion models and ratio estimators
- Guided sampling with adjustable guidance scale
- Quantitative evaluation with matching accuracy metric

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

# InfoNCE (symmetric contrastive learning)
python src/train_ratio.py --loss_type infonce --epochs 30
```

Models will be saved to `checkpoints/ratio/<loss_type>/`.

**⚠️ Important note about InfoNCE:**
- InfoNCE learns the **PMI** (Pointwise Mutual Information): log(q(x,y) / (q(x)q(y))) + const
- This is valid for guidance **only when test marginals = train marginals** (in-domain)
- For out-of-domain guidance (e.g., different diffusion models at test time), use Discriminator/DV/uLSIF/RuLSIF/KLIEP instead
- InfoNCE uses only real pairs (ignores the fake pairs in the dataset)

### 3. Train Classifiers (for Evaluation)

Train classifiers to evaluate guided sampling quality:

```bash
# Train classifiers for both standard and rotated MNIST
python src/train_classifier.py --dataset both --epochs 10
```

Classifiers will be saved to `checkpoints/classifiers/`.

### 4. Generate Samples

```bash
# Unconditional sampling from standard MNIST
python src/sample.py --dataset standard --num_samples 16

# Guided sampling: generate standard MNIST conditioned on rotated MNIST
python src/sample.py --dataset standard --num_samples 16 \
    --guided --loss_type disc --guidance_scale 2.0 \
    --condition_dataset rotated
```

Outputs will be saved to `outputs/`.

### 5. Evaluate Guidance Quality

Compute matching accuracy (% of generated pairs with same digit label):

```bash
# Quick evaluation: single configuration
python src/evaluate_guidance.py --loss_type disc --guidance_scale 2.0 --num_samples 100

# Comprehensive sweep: all methods across multiple scales + plots
python src/run_evaluation_sweep.py
```

The sweep script automatically:
- Tests all trained ratio models
- Evaluates scales from 2.0 to 10.0
- Saves results as JSON
- Generates comparison plots (accuracy vs scale)
- Prints summary table

Results saved to `outputs/evaluation_sweep/` with plots and consolidated JSON.

## Project Structure

```
ratio-guidance-light/
├── data/                    # MNIST data (auto-downloaded)
├── checkpoints/            # Saved models
│   ├── diffusion/         # DDPM checkpoints
│   │   ├── standard/      # Standard MNIST model
│   │   └── rotated/       # Rotated MNIST model
│   ├── ratio/             # Ratio estimator checkpoints
│   │   ├── disc/          # Discriminator loss
│   │   ├── dv/            # Donsker-Varadhan loss
│   │   ├── ulsif/         # uLSIF loss
│   │   ├── rulsif/        # RuLSIF loss
│   │   ├── kliep/         # KLIEP loss
│   │   └── infonce/       # InfoNCE loss (PMI, in-domain only)
│   └── classifiers/       # MNIST classifiers for evaluation
├── outputs/               # Generated samples
├── src/
│   ├── models/
│   │   ├── unet.py       # UNet for DDPM
│   │   ├── ratio_estimator.py  # Density-ratio network
│   │   └── classifier.py       # MNIST classifier for evaluation
│   ├── data/
│   │   └── mnist_dataset.py    # MNIST loaders
│   ├── utils/
│   │   ├── diffusion.py       # Diffusion schedule & sampling
│   │   ├── losses.py          # Ratio losses
│   │   └── trainer.py         # Training utilities
│   ├── train_diffusion.py     # Train DDPM models
│   ├── train_ratio.py         # Train ratio estimators
│   ├── train_classifier.py    # Train classifiers for evaluation
│   ├── sample.py              # Guided sampling
│   ├── visualize_diffusion_samples.py  # Visualize trained models
│   └── evaluate_guidance.py   # Quantitative evaluation
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
  
**Note on Implementation:** This implementation uses a **simplified approach** for generating fake pairs compared to the original MMDisCo paper. Instead of generating fake pairs by running full diffusion sampling from the pre-trained models (which would be `x' ~ p_φ(x)` and `y' ~ p_ψ(y)`), we use **random shuffling of the real dataset** to approximate the product of marginals `p(x) · p(y)`. This is a pragmatic simplification that:
- ✅ Is computationally much faster (no expensive sampling required)
- ✅ Works well when diffusion models are well-trained (as they approximate the true distribution)
- ✅ Is commonly used in practice for educational/simplified implementations
- ⚠️ May differ slightly from the theoretical framework if the pre-trained models have distributional biases

### Loss Functions
| Loss | Objective | Use Case | Notes |
|------|-----------|----------|-------|
| **Discriminator** | Binary classification (q/r) | Most stable, recommended for beginners | General-purpose |
| **DV** | Mutual information lower bound | Theoretical connections to MI | General-purpose |
| **uLSIF** | Least-squares density ratio | Direct ratio estimation | General-purpose |
| **RuLSIF** | Relative uLSIF | More stable with hyperparameter alpha | General-purpose |
| **KLIEP** | KL importance estimation | KL-based objective | General-purpose |
| **InfoNCE** | Symmetric contrastive (PMI) | Contrastive learning approach | ⚠️ **In-domain only** |

**⚠️ InfoNCE Limitation:**
- InfoNCE learns PMI: `log(q(x,y) / (q(x)q(y)))`
- At inference, this gives correct gradient **only if test marginals = train marginals**
- In this MNIST setup, both diffusion models are trained on the same data distribution → InfoNCE is valid
- For out-of-domain guidance or distribution shift, use Discriminator/DV/uLSIF instead (they target `q/r` directly)

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
for loss in disc dv ulsif rulsif kliep infonce; do
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
