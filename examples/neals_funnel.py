"""
Example: SIVI on Neal's Funnel

Neal's Funnel is a classic test for hierarchical models:
    z_1 ~ N(0, 3)
    z_i ~ N(0, exp(z_1)), for i = 2, ..., dim

This creates a funnel-shaped distribution where the variance of z_2:dim
depends on z_1. Standard mean-field VI struggles with this strong dependence.

This example demonstrates:
1. SIVI's ability to handle hierarchical structure
2. Capturing the variance-dependence relationship
3. Comparing performance in different dimensions
"""

import sys
import os

# Add parent directory to path to import sivi package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import numpy as np
from sivi.model import SIVIModel
from sivi.target import NealsFunnel
from sivi.trainer import train, evaluate
from sivi.utils import (
    plot_training_history,
    plot_marginals,
    print_comparison,
    set_random_seed
)


def plot_funnel_structure(model, target, n_samples=2000):
    """
    Plot the funnel structure - z_1 vs z_2 relationship.
    
    The key feature of Neal's funnel is that the spread of z_2:dim
    depends on z_1.
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Target funnel
    axes[0].scatter(z_target[:, 0], z_target[:, 1], alpha=0.3, s=5, c='blue')
    axes[0].set_xlabel('z₁ (controls variance)', fontsize=12)
    axes[0].set_ylabel('z₂', fontsize=12)
    axes[0].set_title('Target: Neal\'s Funnel', fontsize=14)
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-30, 30)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: SIVI approximation
    axes[1].scatter(z_model[:, 0], z_model[:, 1], alpha=0.3, s=5, c='red')
    axes[1].set_xlabel('z₁ (controls variance)', fontsize=12)
    axes[1].set_ylabel('z₂', fontsize=12)
    axes[1].set_title('SIVI Approximation', fontsize=14)
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-30, 30)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Overlay
    axes[2].scatter(z_target[:, 0], z_target[:, 1], alpha=0.2, s=5, c='blue', label='Target')
    axes[2].scatter(z_model[:, 0], z_model[:, 1], alpha=0.2, s=5, c='red', label='SIVI')
    axes[2].set_xlabel('z₁ (controls variance)', fontsize=12)
    axes[2].set_ylabel('z₂', fontsize=12)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].set_xlim(-10, 10)
    axes[2].set_ylim(-30, 30)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    model.train()


def plot_conditional_variance(model, target, n_samples=5000):
    """
    Plot how variance of z_2:dim changes with z_1.
    
    This is the key relationship in Neal's funnel:
    Var(z_i | z_1) = exp(2 * z_1)
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Bin by z_1 and compute variance of z_2 in each bin
    n_bins = 20
    z1_bins = np.linspace(-8, 8, n_bins + 1)
    bin_centers = (z1_bins[:-1] + z1_bins[1:]) / 2
    
    var_target = []
    var_model = []
    
    for i in range(n_bins):
        mask_target = (z_target[:, 0] >= z1_bins[i]) & (z_target[:, 0] < z1_bins[i+1])
        mask_model = (z_model[:, 0] >= z1_bins[i]) & (z_model[:, 0] < z1_bins[i+1])
        
        if mask_target.sum() > 10:
            var_target.append(np.var(z_target[mask_target, 1]))
        else:
            var_target.append(np.nan)
            
        if mask_model.sum() > 10:
            var_model.append(np.var(z_model[mask_model, 1]))
        else:
            var_model.append(np.nan)
    
    # True relationship: Var(z_2 | z_1) = exp(2 * z_1)
    z1_theory = np.linspace(-8, 8, 100)
    var_theory = np.exp(2 * z1_theory)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z1_theory, var_theory, 'k--', linewidth=2, label='True: exp(2·z₁)')
    plt.plot(bin_centers, var_target, 'bo-', linewidth=2, markersize=6, label='Target (empirical)')
    plt.plot(bin_centers, var_model, 'ro-', linewidth=2, markersize=6, label='SIVI (empirical)')
    plt.xlabel('z₁', fontsize=12)
    plt.ylabel('Var(z₂ | z₁)', fontsize=12)
    plt.title('Conditional Variance: The Funnel Relationship', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    model.train()


def main():
    """Run the Neal's Funnel example."""
    
    print("="*70)
    print("SIVI Example: Neal's Funnel")
    print("="*70)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # ==================================================================
    # 1. Setup Target Distribution
    # ==================================================================
    print("\n" + "-"*70)
    print("1. Setting up Neal's Funnel")
    print("-"*70)
    
    dim = 2  # Total dimensions
    target = NealsFunnel(dim=dim, scale=3.0)
    
    print(f" Created Neal's Funnel")
    print(f"   Dimension: {dim}")
    print(f"   Structure: z_1 ~ N(0, 3), z_i ~ N(0, exp(z_1)) for i > 1")

    
    # ==================================================================
    # 2. Setup SIVI Model
    # ==================================================================
    print("\n" + "-"*70)
    print("2. Setting up SIVI model")
    print("-"*70)
    
    latent_dim = dim
    mixing_dim = 16      # Higher for complex dependence structure
    hidden_dim = 128     # Larger network for hierarchical structure
    n_layers = 4

    model = SIVIModel(
        latent_dim=latent_dim,
        mixing_dim=mixing_dim,
        hidden_dim=hidden_dim,
        n_layers= n_layers
    )
    
    print(f" Created SIVI model")
    print(f"  - Latent / Mixing / Hidden dimension: {latent_dim} / {mixing_dim} / {hidden_dim}")
    print(f"  - Number of layers: {n_layers}")
    print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # ==================================================================
    # 3. Train Model
    # ==================================================================
    print("\n" + "-"*70)
    print("3. Training SIVI on Neal's Funnel")
    print("-"*70)
    
    history = train(
        model=model,
        target=target,
        n_iterations=10000,      # More iterations for complex structure
        n_samples=256,          # More samples for stable gradients
        K=20,                   # Higher K for better ELBO estimate
        learning_rate=5e-4,     # Lower learning rate for stability
        print_every=1000,
        early_stop_patience=500,
        early_stop_threshold=1e-4
    )
    
    print(f"\n Training completed")
    print(f"  - Total time: {history['time']:.2f}s")
    print(f"  - Final ELBO: {history['elbo'][-1]:.4f}")
    print(f"  - Converged: {history['converged']}")
    
    # ==================================================================
    # 4. Evaluate Model
    # ==================================================================
    print("\n" + "-"*70)
    print("4. Evaluating model")
    print("-"*70)
    
    eval_results = evaluate(model, target, n_samples=2000, K=15)
    print(f" Evaluation complete")
    print(f"  - ELBO: {eval_results['elbo']:.4f}")
    print(f"  - Mean log p(z): {eval_results['log_p_mean']:.4f}")
    print(f"  - Std log p(z): {eval_results['log_p_std']:.4f}")
    
    # ==================================================================
    # 5. Visualize Results
    # ==================================================================
    print("\n" + "-"*70)
    print("5. Generating visualizations")
    print("-"*70)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Plot funnel structure (z_1 vs z_2)
    print("\nPlotting funnel structure...")
    plot_funnel_structure(model, target, n_samples=2000)
    
    # Plot conditional variance relationship
    print("\nPlotting conditional variance...")
    plot_conditional_variance(model, target, n_samples=5000)
    
    # Plot marginal distributions
    print("\nPlotting marginal distributions (first 6 dimensions)...")
    plot_marginals(model, target, n_samples=2000, n_dims=6)
    
    # ==================================================================
    # 6. Compare Moments
    # ==================================================================
    print("\n" + "-"*70)
    print("6. Comparing moments")
    print("-"*70)
    
    print_comparison(model, target, n_samples=5000)
    
    # ==================================================================
    # 7. Analyze Funnel Capture
    # ==================================================================
    print("\n" + "-"*70)
    print("7. Analyzing funnel capture")
    print("-"*70)
    
    model.eval()
    with torch.no_grad():
        z_target = target.sample(5000).numpy()
        z_model, _ = model.sample(5000)
        z_model = z_model.numpy()
    
    # Check correlation between z_1 and variance of z_2:dim
    def compute_z1_var_correlation(z):
        """Compute how variance of z_2:dim changes with z_1."""
        n_bins = 10
        z1_bins = np.linspace(-6, 6, n_bins + 1)
        bin_centers = (z1_bins[:-1] + z1_bins[1:]) / 2
        variances = []
        
        for i in range(n_bins):
            mask = (z[:, 0] >= z1_bins[i]) & (z[:, 0] < z1_bins[i+1])
            if mask.sum() > 10:
                variances.append(np.var(z[mask, 1:], axis=0).mean())
            else:
                variances.append(np.nan)
        
        return bin_centers, variances
    
    z1_target, var_target = compute_z1_var_correlation(z_target)
    z1_model, var_model = compute_z1_var_correlation(z_model)
    
    print("\nConditional variance pattern:")
    print("  z_1 region  |  Target Var  |  SIVI Var")
    print("  " + "-"*45)
    for i, (z1, vt, vm) in enumerate(zip(z1_target, var_target, var_model)):
        if not np.isnan(vt) and not np.isnan(vm):
            print(f"  {z1:>6.2f}     |  {vt:>10.2f}  |  {vm:>8.2f}")


if __name__ == "__main__":
    main()