"""
Run comprehensive evaluation sweep across guidance scales and plot results.

This script:
1. Evaluates all available ratio models across multiple guidance scales
2. Saves results to JSON files
3. Generates comparison plots

Usage:
    # Evaluate all models with default scales (2.0 to 10.0)
    python src/run_evaluation_sweep.py

    # Custom scales
    python src/run_evaluation_sweep.py --scales 1.0 2.0 3.0 5.0 7.0 10.0

    # Specific loss types only
    python src/run_evaluation_sweep.py --loss_types disc rulsif

    # More samples for better statistics
    python src/run_evaluation_sweep.py --num_samples 200
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evaluate_guidance import evaluate_guidance


def run_evaluation_sweep(
    loss_types=None,
    guidance_scales=None,
    num_samples=100,
    device='cuda',
    output_dir='outputs/evaluation_sweep'
):
    """
    Run evaluation across multiple loss types and guidance scales.

    Args:
        loss_types: List of loss types to evaluate (None = auto-detect)
        guidance_scales: List of guidance scales to test
        num_samples: Number of samples per evaluation
        device: Device to use
        output_dir: Directory to save results

    Returns:
        dict: Results organized by loss_type -> scale -> accuracy
    """
    # Default scales if not specified
    if guidance_scales is None:
        guidance_scales = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Auto-detect available loss types if not specified
    if loss_types is None:
        all_loss_types = ['disc', 'dv', 'ulsif', 'rulsif', 'kliep', 'infonce']
        ratio_dir = Path('checkpoints/ratio')
        loss_types = []
        for loss_type in all_loss_types:
            if (ratio_dir / loss_type / 'best_model.pt').exists():
                loss_types.append(loss_type)

        if not loss_types:
            raise FileNotFoundError(
                "No trained ratio models found in checkpoints/ratio/\n"
                "Please train at least one model first."
            )

    print(f"\n{'='*70}")
    print(f"EVALUATION SWEEP")
    print(f"{'='*70}")
    print(f"Loss types: {loss_types}")
    print(f"Guidance scales: {guidance_scales}")
    print(f"Samples per eval: {num_samples}")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Store all results
    all_results = {}

    # Run evaluations
    total_evals = len(loss_types) * len(guidance_scales)
    eval_count = 0

    for loss_type in loss_types:
        print(f"\n{'='*70}")
        print(f"Evaluating {loss_type.upper()} loss")
        print(f"{'='*70}")

        all_results[loss_type] = {}

        for scale in guidance_scales:
            eval_count += 1
            print(f"\n[{eval_count}/{total_evals}] {loss_type} @ scale={scale}")

            try:
                # Run evaluation
                results = evaluate_guidance(
                    loss_type=loss_type,
                    guidance_scale=scale,
                    num_samples=num_samples,
                    device=device,
                    save_results=True,
                    output_dir=str(output_path / 'individual_results')
                )

                # Store accuracy
                all_results[loss_type][scale] = results['accuracy']

                print(f"  ✓ Accuracy: {results['accuracy']:.2f}%")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_results[loss_type][scale] = None

    # Save consolidated results
    results_file = output_path / 'sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved consolidated results to {results_file}")

    return all_results


def plot_results(results, output_dir='outputs/evaluation_sweep'):
    """
    Create comparison plots from evaluation results.

    Args:
        results: Dictionary of {loss_type: {scale: accuracy}}
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    loss_types = list(results.keys())

    # Get all scales (should be same for all loss types)
    all_scales = set()
    for loss_data in results.values():
        all_scales.update(loss_data.keys())
    scales = sorted(all_scales)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Color map for different loss types
    colors = {
        'disc': '#1f77b4',      # blue
        'dv': '#ff7f0e',        # orange
        'ulsif': '#2ca02c',     # green
        'rulsif': '#d62728',    # red
        'kliep': '#9467bd',     # purple
        'infonce': '#8c564b',   # brown
    }

    # Marker styles
    markers = {
        'disc': 'o',
        'dv': 's',
        'ulsif': '^',
        'rulsif': 'D',
        'kliep': 'v',
        'infonce': 'p',         # pentagon
    }

    # Plot each loss type
    for loss_type in loss_types:
        # Extract accuracies for this loss type
        accuracies = []
        valid_scales = []

        for scale in scales:
            acc = results[loss_type].get(scale)
            if acc is not None:
                valid_scales.append(scale)
                accuracies.append(acc)

        if accuracies:
            plt.plot(
                valid_scales,
                accuracies,
                marker=markers.get(loss_type, 'o'),
                color=colors.get(loss_type, 'gray'),
                linewidth=2,
                markersize=8,
                label=loss_type.upper(),
                alpha=0.8
            )

    # Formatting
    plt.xlabel('Guidance Scale', fontsize=14, fontweight='bold')
    plt.ylabel('Matching Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Guidance Quality vs. Guidance Scale\nAcross Different Ratio Estimation Methods',
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)

    # Add reference lines
    plt.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='Random baseline (10%)')
    plt.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='Excellent (90%)')

    # Set y-axis limits
    plt.ylim(0, 100)

    # Set x-axis to show all scales
    plt.xticks(scales)

    plt.tight_layout()

    # Save figure
    plot_file = output_path / 'accuracy_vs_scale.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_file}")

    # Also save as PDF for publication quality
    pdf_file = output_path / 'accuracy_vs_scale.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ Saved PDF to {pdf_file}")

    plt.close()

    # Create a second plot: differences from best method
    create_comparison_plot(results, output_path)


def create_comparison_plot(results, output_path):
    """Create a plot showing performance relative to best method at each scale."""

    # Get scales
    all_scales = set()
    for loss_data in results.values():
        all_scales.update(loss_data.keys())
    scales = sorted(all_scales)

    # Find best accuracy at each scale
    best_at_scale = {}
    for scale in scales:
        max_acc = 0
        for loss_type, loss_data in results.items():
            acc = loss_data.get(scale)
            if acc is not None and acc > max_acc:
                max_acc = acc
        best_at_scale[scale] = max_acc

    # Create figure
    plt.figure(figsize=(12, 8))

    colors = {
        'disc': '#1f77b4',
        'dv': '#ff7f0e',
        'ulsif': '#2ca02c',
        'rulsif': '#d62728',
        'kliep': '#9467bd',
        'infonce': '#8c564b',
    }

    for loss_type, loss_data in results.items():
        differences = []
        valid_scales = []

        for scale in scales:
            acc = loss_data.get(scale)
            if acc is not None:
                diff = acc - best_at_scale[scale]
                valid_scales.append(scale)
                differences.append(diff)

        if differences:
            plt.plot(
                valid_scales,
                differences,
                marker='o',
                color=colors.get(loss_type, 'gray'),
                linewidth=2,
                markersize=8,
                label=loss_type.upper(),
                alpha=0.8
            )

    plt.xlabel('Guidance Scale', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Difference from Best (%)', fontsize=14, fontweight='bold')
    plt.title('Performance Gap Relative to Best Method\n(Negative = Worse than Best)',
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xticks(scales)
    plt.tight_layout()

    comparison_file = output_path / 'relative_performance.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {comparison_file}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    # Get scales
    all_scales = set()
    for loss_data in results.values():
        all_scales.update(loss_data.keys())
    scales = sorted(all_scales)

    # Print header
    header = f"{'Loss Type':<12}"
    for scale in scales:
        header += f"{scale:>8.1f}"
    header += f"{'  Best':>8}"
    print(header)
    print("-" * len(header))

    # Print each loss type
    for loss_type, loss_data in sorted(results.items()):
        row = f"{loss_type:<12}"
        best_acc = 0

        for scale in scales:
            acc = loss_data.get(scale)
            if acc is not None:
                row += f"{acc:>8.2f}"
                best_acc = max(best_acc, acc)
            else:
                row += f"{'N/A':>8}"

        row += f"{best_acc:>8.2f}"
        print(row)

    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run comprehensive evaluation sweep across guidance scales'
    )
    parser.add_argument('--loss_types', type=str, nargs='+',
                       choices=['disc', 'dv', 'ulsif', 'rulsif', 'kliep', 'infonce'],
                       help='Loss types to evaluate (default: auto-detect all)')
    parser.add_argument('--scales', type=float, nargs='+',
                       help='Guidance scales to test (default: 2.0 to 10.0)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples per evaluation (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation_sweep',
                       help='Output directory (default: outputs/evaluation_sweep)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting (only run evaluations)')

    args = parser.parse_args()

    # Run evaluation sweep
    results = run_evaluation_sweep(
        loss_types=args.loss_types,
        guidance_scales=args.scales,
        num_samples=args.num_samples,
        device=args.device,
        output_dir=args.output_dir
    )

    # Print summary
    print_summary_table(results)

    # Create plots
    if not args.no_plot:
        print(f"\n{'='*70}")
        print("Creating plots...")
        print(f"{'='*70}\n")
        plot_results(results, args.output_dir)

    print(f"\n{'='*70}")
    print("EVALUATION SWEEP COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")
