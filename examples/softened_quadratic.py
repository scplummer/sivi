"""
Example: SIVI on Softened Quadratic Distribution

The softened quadratic tests whether bounded Hessian is sufficient for SIVI success.

Structure:
    z_1 ~ N(0, 1)
    z_2 = tanh(c * z_1^2) + ε, where ε ~ N(0, σ^2)

The key insight:
- Hard quadratic (z_2 = z_1^2) has unbounded second derivatives → SIVI might fail
- Softened version (z_2 = tanh(c * z_1^2)) has bounded derivatives → should work

The parameter c controls curvature:
    c → 0: Nearly linear (easy)
    c = 1: Moderate curvature
    c → ∞: Approaches hard quadratic (hard)

This tests the theoretical prediction that bounded Hessian enables SIVI to capture
nonlinear dependencies.
"""

import sys
import os

# Add parent directory to path to import sivi package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import numpy as np
from sivi.model import SIVIModel
from sivi.target import SoftenedQuadratic
from sivi.trainer import train, evaluate
from sivi.utils import (
    plot_training_history,
    print_comparison,
    set_random_seed
)


def plot_relationship(model, target, n_samples=2000):
    """
    Plot the z_1 vs z_2 relationship to see if SIVI captured the nonlinearity.
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # True relationship
    z1_grid = np.linspace(-3, 3, 100)
    z2_true = np.tanh(target.c * z1_grid ** 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Target samples
    axes[0].scatter(z_target[:, 0], z_target[:, 1], alpha=0.4, s=10, c='blue')
    axes[0].plot(z1_grid, z2_true, 'k-', linewidth=3, label=f'True: tanh({target.c}·z₁²)')
    axes[0].set_xlabel('z₁', fontsize=12)
    axes[0].set_ylabel('z₂', fontsize=12)
    axes[0].set_title('Target Distribution', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-0.5, 1.5)
    
    # Panel 2: SIVI samples
    axes[1].scatter(z_model[:, 0], z_model[:, 1], alpha=0.4, s=10, c='red')
    axes[1].plot(z1_grid, z2_true, 'k-', linewidth=3, label=f'True: tanh({target.c}·z₁²)')
    axes[1].set_xlabel('z₁', fontsize=12)
    axes[1].set_ylabel('z₂', fontsize=12)
    axes[1].set_title('SIVI Approximation', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-0.5, 1.5)
    
    # Panel 3: Overlay
    axes[2].scatter(z_target[:, 0], z_target[:, 1], alpha=0.3, s=5, c='blue', label='Target')
    axes[2].scatter(z_model[:, 0], z_model[:, 1], alpha=0.3, s=5, c='red', label='SIVI')
    axes[2].plot(z1_grid, z2_true, 'k-', linewidth=3, label=f'True curve')
    axes[2].set_xlabel('z₁', fontsize=12)
    axes[2].set_ylabel('z₂', fontsize=12)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    model.train()


def plot_conditional_mean(model, target, n_samples=5000):
    """
    Plot E[z_2 | z_1] to see if SIVI learned the functional relationship.
    
    The true relationship is: E[z_2 | z_1] = tanh(c * z_1^2)
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Bin by z_1 and compute mean of z_2 in each bin
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
    
    # True relationship
    z1_theory = np.linspace(-3, 3, 100)
    z2_theory = np.tanh(target.c * z1_theory ** 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z1_theory, z2_theory, 'k-', linewidth=3, label=f'True: tanh({target.c}·z₁²)')
    plt.plot(bin_centers, mean_target, 'bo-', linewidth=2, markersize=6, label='Target (empirical)', alpha=0.7)
    plt.plot(bin_centers, mean_model, 'ro-', linewidth=2, markersize=6, label='SIVI (empirical)', alpha=0.7)
    plt.xlabel('z₁', fontsize=12)
    plt.ylabel('E[z₂ | z₁]', fontsize=12)
    plt.title('Conditional Mean: Did SIVI Learn the Nonlinear Relationship?', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.tight_layout()
    plt.show()
    
    model.train()


def compute_curvature_match(model, target, n_samples=5000):
    """
    Quantify how well SIVI captured the curvature.
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Bin by z_1 and compute mean z_2
    n_bins = 15
    z1_bins = np.linspace(-2.5, 2.5, n_bins + 1)
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
    
    # True curve at bin centers
    true_curve = np.tanh(target.c * bin_centers ** 2)
    
    # Remove NaNs
    valid = ~(np.isnan(mean_target) | np.isnan(mean_model))
    mean_target = np.array(mean_target)[valid]
    mean_model = np.array(mean_model)[valid]
    true_curve = true_curve[valid]
    bin_centers = bin_centers[valid]
    
    # Compute errors
    target_error = np.mean((mean_target - true_curve) ** 2)
    model_error = np.mean((mean_model - true_curve) ** 2)
    
    return {
        'target_mse': target_error,
        'model_mse': model_error,
        'bin_centers': bin_centers,
        'mean_target': mean_target,
        'mean_model': mean_model,
        'true_curve': true_curve
    }


def main():
    """Run the softened quadratic example."""
    
    print("\n" + "="*70)
    print("Testing SIVI on Softened Quadratic (Curvature Test)")
    print("="*70 + "\n")
    
    set_random_seed(42)
    
    # Setup target
    print("Setting up softened quadratic distribution...")
    c = 0.5      # Curvature parameter (start moderate)
    sigma = 0.3  # Noise level (higher for easier learning)
    linear_test = False  # Set to True to test linear relationship
    
    target = SoftenedQuadratic(c=c, sigma=sigma, linear_test=linear_test)
    
    if linear_test:
        print(f"  LINEAR TEST MODE")
        print(f"  z_1 ~ N(0, 1)")
        print(f"  z_2 = {c} * z_1 + N(0, {sigma}^2)")
        print(f"  Testing if SIVI can learn simple linear dependence")
    else:
        print(f"  z_1 ~ N(0, 1)")
        print(f"  z_2 = tanh({c} * z_1^2) + N(0, {sigma}^2)")
        print(f"  This has bounded second derivatives (unlike z_2 = z_1^2)")
    
    z_sample = target.sample(1000)
    corr = np.corrcoef(z_sample.numpy().T)[0, 1]
    print(f"\nSample correlation: {corr:.3f}")
    if linear_test:
        print("(Should be high for linear relationship)")
    else:
        print("(Nonlinear but monotonic relationship)")
    
    print("\nInitializing SIVI model (aggressive settings + full covariance)...")
    model = SIVIModel(
        latent_dim=2,
        mixing_dim=16,       # Very high - gives network flexibility
        hidden_dim=128,      # Large network
        n_layers=4,          # Deep network
        full_covariance=True # Allows q(z|ε) to have correlated components
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {n_params:,} parameters")
    print(f"  Using mixing_dim=32 (high for capturing dependencies)")
    print(f"  Using FULL covariance in q(z|ε) - allows z_1 and z_2 to be correlated!")
    print(f"  This should help capture the z_2 = f(z_1) relationship")
    print(f"  Computationally expensive but necessary for nonlinear relationships")
    
    # Train - AGGRESSIVE
    print("\nTraining (this will take several minutes)...")
    print("Note: Nonlinear dependencies are challenging for SIVI\n")
    
    history = train(
        model=model,
        target=target,
        n_iterations=5000,     # Much longer
        n_samples=1000,         # Many samples per iteration
        K=20,                   # High importance samples
        learning_rate=1e-3,     # Standard rate
        print_every=1000,
        early_stop_patience=None  # Don't stop early
    )
    
    print(f"\nTraining finished in {history['time']:.1f}s")
    print(f"Final ELBO: {history['elbo'][-1]:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate(model, target, n_samples=1000, K=20)
    print(f"  ELBO: {eval_results['elbo']:.4f}")
    print(f"  Mean log p(z): {eval_results['log_p_mean']:.4f}")
    
    # Visualize
    print("\nGenerating plots...")
    print("  (1/3) Training curve")
    plot_training_history(history)
    
    print("  (2/3) Scatter plots showing the relationship")
    plot_relationship(model, target, n_samples=2000)
    
    print("  (3/3) Conditional mean E[z_2 | z_1]")
    plot_conditional_mean(model, target, n_samples=5000)
    
    # Moment comparison
    print("\nComparing moments:")
    print_comparison(model, target, n_samples=5000)
    
    # Curvature analysis
    print("Analyzing how well SIVI captured the nonlinear relationship...")
    curvature_results = compute_curvature_match(model, target, n_samples=5000)
    
    print(f"\nMean squared error in E[z_2 | z_1]:")
    print(f"  Target (sampling error): {curvature_results['target_mse']:.6f}")
    print(f"  SIVI approximation:      {curvature_results['model_mse']:.6f}")
    
    if curvature_results['model_mse'] < 0.01:
        print("  ✓ Excellent capture of the nonlinear relationship!")
    elif curvature_results['model_mse'] < 0.05:
        print("  ✓ Good capture of the nonlinear relationship")
    else:
        print("  ✗ SIVI struggled to capture the curvature")
    
    # Check correlation
    model.eval()
    with torch.no_grad():
        z_model, _ = model.sample(5000)
        z_model = z_model.numpy()
    
    corr_model = np.corrcoef(z_model.T)[0, 1]
    print(f"\nCorrelation check:")
    print(f"  Target: {corr:.3f}")
    print(f"  SIVI:   {corr_model:.3f}")
    
    print("\n" + "="*70)
    print("Done! Key question: Did SIVI capture the nonlinear relationship?")
    print("Check the conditional mean plot - it should match the black curve.")
    print("\nInterpretation:")
    if curvature_results['model_mse'] < 0.01:
        print("  ✓ MSE < 0.01: SIVI successfully learned the relationship!")
    elif curvature_results['model_mse'] < 0.05:
        print("  ~ MSE < 0.05: SIVI captured the general shape")
    else:
        print("  ✗ MSE > 0.05: SIVI struggled with this nonlinearity")
if __name__ == "__main__":
    main()