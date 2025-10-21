# Evaluation Guide

This guide explains how to quantitatively evaluate the quality of guided sampling.

## Overview

The evaluation measures **matching accuracy**: the percentage of generated pairs where the standard MNIST digit has the same label as the rotated condition digit.

**Perfect guidance** would achieve 100% accuracy (every generated digit matches its condition).

## Setup

### 1. Train Classifiers

First, train classifiers to recognize digits in both standard and rotated forms:

```bash
python src/train_classifier.py --dataset both --epochs 10
```

This trains two classifiers:
- `checkpoints/classifiers/standard_classifier.pt` - recognizes normal MNIST
- `checkpoints/classifiers/rotated_classifier.pt` - recognizes 90° rotated MNIST

Expected accuracy: **~99%** on test set (MNIST is easy!)

## Running Evaluations

### Evaluate Single Configuration

```bash
python src/evaluate_guidance.py \
    --loss_type disc \
    --guidance_scale 2.0 \
    --num_samples 100
```

**Output:**
- Matching accuracy (e.g., "85.0%")
- JSON results: `outputs/evaluation/disc_scale2.0_results.json`
- Sample visualization: `outputs/evaluation/disc_scale2.0_samples.png`

### Compare Guidance Scales

Test how guidance strength affects accuracy:

```bash
python src/evaluate_guidance.py \
    --loss_type disc \
    --scales 0.0 0.5 1.0 2.0 5.0 \
    --num_samples 100
```

**Expected behavior:**
- Scale 0.0: ~10% (random baseline)
- Scale 0.5-1.0: 40-60% (weak guidance)
- Scale 2.0: 70-85% (good guidance)
- Scale 5.0+: 80-90% (strong guidance, may have artifacts)

### Compare All Loss Functions

```bash
python src/evaluate_guidance.py \
    --eval_all \
    --guidance_scale 2.0 \
    --num_samples 100
```

Compares all trained ratio estimators at the same guidance scale.

## Interpreting Results

### Matching Accuracy

- **10%**: Random baseline (no guidance)
- **50%**: Weak correlation
- **70-80%**: Good guidance
- **85%+**: Excellent guidance
- **95%+**: Near-perfect (rarely achieved)

### Common Issues

**Low accuracy (<40%):**
- Ratio model not well-trained
- Guidance scale too low
- Diffusion models not converged

**High variance:**
- Need more evaluation samples (increase `--num_samples`)
- Classifiers may be overfitting

**Accuracy decreasing at high scales:**
- Over-guidance causing artifacts
- Ratio estimator becoming unreliable

## Comprehensive Evaluation Sweep

For systematic evaluation across all methods and scales, use the automated sweep script:

```bash
# Run full evaluation sweep (scales 2.0 to 10.0)
python src/run_evaluation_sweep.py

# Custom scales
python src/run_evaluation_sweep.py --scales 1.0 2.0 3.0 5.0 7.0 10.0

# More samples for better statistics
python src/run_evaluation_sweep.py --num_samples 200

# Specific loss types only
python src/run_evaluation_sweep.py --loss_types disc rulsif
```

This script automatically:
- Evaluates all available ratio models
- Tests multiple guidance scales
- Saves results to JSON
- Generates comparison plots
- Prints summary table

## Output Files

### Evaluation Sweep Outputs

After running `run_evaluation_sweep.py`, you'll get:

```
outputs/evaluation_sweep/
├── sweep_results.json              # Consolidated results for all methods/scales
├── accuracy_vs_scale.png           # Main comparison plot
├── accuracy_vs_scale.pdf           # Publication-quality version
├── relative_performance.png        # Performance gap from best method
└── individual_results/             # Detailed results per configuration
    ├── disc_scale2.0_results.json
    ├── disc_scale2.0_samples.png
    ├── disc_scale5.0_results.json
    └── ...
```

### JSON Results Format

Individual result file:
```json
{
  "loss_type": "disc",
  "guidance_scale": 2.0,
  "num_samples": 100,
  "accuracy": 82.5,
  "num_matches": 82,
  "total_samples": 100
}
```

Consolidated sweep results:
```json
{
  "disc": {
    "2.0": 75.5,
    "3.0": 80.2,
    "5.0": 85.0,
    "10.0": 87.5
  },
  "rulsif": {
    "2.0": 73.0,
    "3.0": 78.5,
    "5.0": 82.0,
    "10.0": 84.5
  }
}
```

### Visualization

Sample images saved as grids:
- Row pattern: [condition, generated, condition, generated, ...]
- Top: Rotated condition (what we gave as input)
- Bottom: Standard generated (what the model produced)

Look for:
- ✅ Same digit shapes between pairs
- ✅ Correct rotation alignment
- ⚠️ Mismatches (different digits)
- ⚠️ Artifacts or blurriness

## Expected Results

Based on the implementation, you should see:

| Loss Type | Scale 0.5 | Scale 1.0 | Scale 2.0 | Scale 5.0 |
|-----------|-----------|-----------|-----------|-----------|
| **Discriminator** | 45-55% | 65-75% | 75-85% | 80-90% |
| **DV** | 40-50% | 60-70% | 70-80% | 75-85% |
| **uLSIF** | 40-50% | 60-70% | 70-80% | 75-85% |
| **RuLSIF** | 45-55% | 65-75% | 75-85% | 80-90% |
| **KLIEP** | 40-50% | 60-70% | 70-80% | 75-85% |

*Note: Actual results depend on training quality and random variation.*

## Troubleshooting

### Classifiers not found

```bash
python src/train_classifier.py --dataset both --epochs 10
```

### Ratio models not found

```bash
python src/train_ratio.py --loss_type disc --epochs 30
```

### Out of memory

Reduce batch size in evaluation:
```python
# Edit src/evaluate_guidance.py, line ~200
batch_size = 8  # Reduce from 16
```

### Slow evaluation

- Reduce `--num_samples` (100 is usually sufficient)
- Use CPU if GPU is unavailable (slower but works)

## Advanced Usage

### Custom Classifier

To use your own classifier:

```python
from models.classifier import MNISTClassifier

# Load custom classifier
classifier = MNISTClassifier()
checkpoint = torch.load('path/to/classifier.pt')
classifier.load_state_dict(checkpoint['model_state_dict'])
```

### Save All Samples

Modify `evaluate_guidance.py` to save all generated samples (not just first 16):

```python
# In evaluate_guidance function, after generating all samples
for i in range(len(generated_imgs_all)):
    vutils.save_image(
        generated_imgs_all[i:i+1],
        output_path / f'sample_{i:04d}.png',
        normalize=True,
        value_range=(-1, 1)
    )
```

## Citation

If you use this evaluation framework, please acknowledge the use of classifier-based matching accuracy as a metric for conditional generation quality.
