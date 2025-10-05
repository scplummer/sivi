"""
Example: SIVI on Warped Gaussian Distributions

Tests SIVI on various functional relationships:
    z_1 ~ N(0, 1)
    z_2 = f(z_1) + ε, where ε ~ N(0, σ^2)

Different warping functions test different aspects:
- Polynomial (cubic): Tests if SIVI can capture polynomial dependencies
- Sine: Tests periodic/oscillating relationships  
- Tanh: Tests bounded nonlinear relationships (similar to softened quadratic)

This explores the limits of what functional forms SIVI can learn.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import numpy as np
from sivi.model import SIVIModel
from sivi.target import WarpedGaussian
from sivi.trainer import train, evaluate
from sivi.utils import (
    plot_training_history,
    print_comparison,
    set_random_seed
)


def plot_warp_relationship(model, target, n_samples=2000):
    """
    Plot the z_1 vs z_2 relationship.
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # True relationship
    z1_grid = np.linspace(-3, 3, 100)
    z1_tensor = torch.tensor(z1_grid, dtype=torch.float32)
    z2_true = target._warp_function(z1_tensor).numpy()
    
    # Get warp description
    if target.warp_type == 'polynomial':
        poly_str = ' + '.join([f'{c:.2f}z₁^{i}' if i > 0 else f'{c:.2f}' 
                               for i, c in enumerate(target.coeffs) if c != 0])
        warp_label = f'f(z₁) = {poly_str}'
    elif target.warp_type == 'sin':
        warp_label = f'f(z₁) = sin({target.scale}·z₁)'
    elif target.warp_type == 'tanh':
        warp_label = f'f(z₁) = tanh({target.scale}·z₁)'
    else:
        warp_label = f'f(z₁) = {target.warp_type}({target.scale}·z₁)'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Target
    axes[0].scatter(z_target[:, 0], z_target[:, 1], alpha=0.4, s=10, c='blue')
    axes[0].plot(z1_grid, z2_true, 'k-', linewidth=3, label=warp_label)
    axes[0].set_xlabel('z₁', fontsize=12)
    axes[0].set_ylabel('z₂', fontsize=12)
    axes[0].set_title('Target Distribution', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-3, 3)
    
    # Panel 2: SIVI
    axes[1].scatter(z_model[:, 0], z_model[:, 1], alpha=0.4, s=10, c='red')
    axes[1].plot(z1_grid, z2_true, 'k-', linewidth=3, label=warp_label)
    axes[1].set_xlabel('z₁', fontsize=12)
    axes[1].set_ylabel('z₂', fontsize=12)
    axes[1].set_title('SIVI Approximation', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-3, 3)
    
    # Panel 3: Overlay
    axes[2].scatter(z_target[:, 0], z_target[:, 1], alpha=0.3, s=5, c='blue', label='Target')
    axes[2].scatter(z_model[:, 0], z_model[:, 1], alpha=0.3, s=5, c='red', label='SIVI')
    axes[2].plot(z1_grid, z2_true, 'k-', linewidth=3, label='True curve')
    axes[2].set_xlabel('z₁', fontsize=12)
    axes[2].set_ylabel('z₂', fontsize=12)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-3, 3)
    
    plt.tight_layout()
    plt.show()
    
    model.train()


def plot_conditional_mean(model, target, n_samples=5000):
    """
    Plot E[z_2 | z_1] to verify functional form learning.
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Bin by z_1
    n_bins = 25
    z1_bins = np.linspace(-3, 3, n_bins + 1)
    bin_centers = (z1_bins[:-1] + z1_bins[1:]) / 2
    
    mean_target = []
    mean_model = []
    
    for i in range(n_bins):
        mask_target = (z_target[:, 0] >= z1_bins[i]) & (z_target[:, 0] < z1_bins[i+1])
        mask_model = (z_model[:, 0] >= z1_bins[i]) & (z_model[:, 0] < z1_bins[i+1])
        
        if mask_target.sum() > 10:
            mean_target.append(np.mean(z_target[mask_target, 1]))
        else:
            mean_target.append(np.nan)
            
        if mask_model.sum() > 10:
            mean_model.append(np.mean(z_model[mask_model, 1]))
        else:
            mean_model.append(np.nan)
    
    # True curve
    z1_theory = np.linspace(-3, 3, 100)
    z1_tensor = torch.tensor(z1_theory, dtype=torch.float32)
    z2_theory = target._warp_function(z1_tensor).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(z1_theory, z2_theory, 'k-', linewidth=3, label='True f(z₁)')
    plt.plot(bin_centers, mean_target, 'bo-', linewidth=2, markersize=6, label='Target (empirical)', alpha=0.7)
    plt.plot(bin_centers, mean_model, 'ro-', linewidth=2, markersize=6, label='SIVI (empirical)', alpha=0.7)
    plt.xlabel('z₁', fontsize=12)
    plt.ylabel('E[z₂ | z₁]', fontsize=12)
    plt.title(f'Conditional Mean: {target.warp_type.capitalize()} Warp', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.tight_layout()
    plt.show()
    
    model.train()


def compute_warp_mse(model, target, n_samples=5000):
    """
    Compute MSE in learning the warping function.
    """
    model.eval()
    
    with torch.no_grad():
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Bin by z_1
    n_bins = 20
    z1_bins = np.linspace(-2.5, 2.5, n_bins + 1)
    bin_centers = (z1_bins[:-1] + z1_bins[1:]) / 2
    
    mean_model = []
    for i in range(n_bins):
        mask = (z_model[:, 0] >= z1_bins[i]) & (z_model[:, 0] < z1_bins[i+1])
        if mask.sum() > 10:
            mean_model.append(np.mean(z_model[mask, 1]))
        else:
            mean_model.append(np.nan)
    
    # True curve at bin centers
    z1_tensor = torch.tensor(bin_centers, dtype=torch.float32)
    true_curve = target._warp_function(z1_tensor).numpy()
    
    # Remove NaNs
    valid = ~np.isnan(mean_model)
    mean_model = np.array(mean_model)[valid]
    true_curve = true_curve[valid]
    
    # Compute MSE
    mse = np.mean((mean_model - true_curve) ** 2)
    
    return mse


def main():
    """Run warped Gaussian examples."""
    
    print("\n" + "="*70)
    print("Testing SIVI on Warped Gaussian Distributions")
    print("="*70 + "\n")
    
    set_random_seed(42)
    
    # Choose which warp to test
    print("Available warps:")
    print("  1. Cubic polynomial: z_2 = 0.3z_1 + 0.2z_1² + 0.1z_1³")
    print("  2. Sine wave: z_2 = sin(2z_1)")
    print("  3. Tanh: z_2 = tanh(1.5z_1)")
    
    # Start with cubic (easiest to visualize)
    warp_choice = 'cubic'  # Change to 'sine' or 'tanh' to test others
    
    if warp_choice == 'cubic':
        target = WarpedGaussian.create_cubic(sigma=0.2)
        print("\nTesting: Cubic polynomial")
    elif warp_choice == 'sine':
        target = WarpedGaussian.create_sine(scale=2.0, sigma=0.2)
        print("\nTesting: Sine wave")
    else:  # Default to tanh
        target = WarpedGaussian.create_tanh(scale=1.5, sigma=0.2)
        print("\nTesting: Tanh")
    
    print(f"  Noise level: σ = {target.sigma}")
    
    z_sample = target.sample(1000)
    corr = np.corrcoef(z_sample.numpy().T)[0, 1]
    print(f"  Sample correlation: {corr:.3f}")
    
    # Setup model with full covariance
    print("\nInitializing SIVI model with full covariance...")
    model = SIVIModel(
        latent_dim=2,
        mixing_dim=16,
        hidden_dim=128,
        n_layers=4,
        full_covariance=True
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {n_params:,} parameters")
    print(f"  Full covariance enabled (critical for learning dependencies)")
    
    # Train
    print("\nTraining...\n")
    
    history = train(
        model=model,
        target=target,
        n_iterations=5000,
        n_samples=1000,
        K=20,
        learning_rate=1e-3,
        print_every=2000,
        early_stop_patience=None
    )
    
    print(f"\nTraining finished in {history['time']:.1f}s")
    print(f"Final ELBO: {history['elbo'][-1]:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate(model, target, n_samples=2000, K=15)
    print(f"  ELBO: {eval_results['elbo']:.4f}")
    print(f"  Mean log p(z): {eval_results['log_p_mean']:.4f}")
    
    # Visualize
    print("\nGenerating plots...")
    print("  (1/3) Training curve")
    plot_training_history(history)
    
    print("  (2/3) Warp relationship")
    plot_warp_relationship(model, target, n_samples=2000)
    
    print("  (3/3) Conditional mean")
    plot_conditional_mean(model, target, n_samples=5000)
    
    # Moment comparison
    print("\nComparing moments:")
    print_comparison(model, target, n_samples=5000)
    
    # Compute MSE
    mse = compute_warp_mse(model, target, n_samples=5000)
    print(f"\nMSE in learning warp function: {mse:.6f}")
    
    if mse < 0.01:
        print("  ✓ Excellent capture of the warping function!")
    elif mse < 0.05:
        print("  ✓ Good capture of the warping function")
    else:
        print("  ~ Moderate capture - some deviation from true function")
    
    # Check correlation
    model.eval()
    with torch.no_grad():
        z_model, _ = model.sample(5000)
        z_model = z_model.numpy()
    corr_model = np.corrcoef(z_model.T)[0, 1]
    
    print(f"\nCorrelation check:")
    print(f"  Target: {corr:.3f}")
    print(f"  SIVI:   {corr_model:.3f}")


if __name__ == "__main__":
    main()