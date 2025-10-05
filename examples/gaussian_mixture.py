"""
Example: SIVI on 2D Gaussian Mixture

This example demonstrates:
1. Setting up a bimodal Gaussian mixture target
2. Training SIVI to approximate it
3. Visualizing the results
4. Comparing moments
"""

import sys
import os

# Add parent directory to path to import sivi package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from sivi.model import SIVIModel
from sivi.target import GaussianMixture
from sivi.trainer import train, evaluate
from sivi.utils import (
    plot_training_history,
    plot_2d_comparison,
    plot_2d_density_comparison,
    plot_marginals,
    print_comparison,
    set_random_seed
)


def main():
    """Run the Gaussian mixture example."""
    
    print("="*70)
    print("SIVI Example: 2D Gaussian Mixture")
    print("="*70)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    print("\n Random seed set to 42")
    
    # ==================================================================
    # 1. Setup Target Distribution
    # ==================================================================
    print("\n" + "-"*70)
    print("1. Setting up target distribution")
    print("-"*70)
    
    # Create a bimodal Gaussian mixture
    # Two modes separated by distance 4.0
    target = GaussianMixture.create_2d_bimodal(separation=4.0)
    print(f" Created bimodal Gaussian mixture")
    print(f"  - Number of components: {target.n_components}")
    print(f"  - Dimension: {target.dim}")
    print(f"  - Mode separation: 4.0")
    
    # Sample from target to visualize
    z_target_sample = target.sample(100)
    print(f" Sampled {z_target_sample.shape[0]} points from target")
    
    # ==================================================================
    # 2. Setup SIVI Model
    # ==================================================================
    print("\n" + "-"*70)
    print("2. Setting up SIVI model")
    print("-"*70)

    #set_random_seed(123)
    
    latent_dim = 2      # Must match target dimension
    mixing_dim = 20     # Higher = more flexible, but harder to train
    hidden_dim = 128    # Size of neural network hidden layers
    
    model = SIVIModel(
        latent_dim=latent_dim,
        mixing_dim=mixing_dim,
        hidden_dim=hidden_dim,
        n_layers=4 # number hidden layers
    )
    
    print(f" Created SIVI model")
    print(f"  - Latent / Mixing / Hidden dimension: {latent_dim} / {mixing_dim} / {hidden_dim}")
    print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # ==================================================================
    # 3. Train Model
    # ==================================================================
    print("\n" + "-"*70)
    print("3. Training SIVI")
    print("-"*70)
    
    history = train(
        model=model,
        target=target,
        n_iterations=5000,
        n_samples=256,   # Samples per iteration
        K=20,   # Importance samples for ELBO, Higher = More accurate but slower
        learning_rate=5e-4,
        print_every=500,
        early_stop_patience=1000,
        early_stop_threshold=1e-3
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
    
    eval_results = evaluate(model, target, n_samples=2000, K=20)
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
    
    # Plot 2D comparison
    print("\nPlotting 2D sample comparison...")
    plot_2d_comparison(
        model, 
        target, 
        n_samples=1000,
        xlim=(-6, 6),
        ylim=(-6, 6)
    )
    
    # Plot density comparison
    print("\nPlotting density comparison...")
    plot_2d_density_comparison(
        model,
        target,
        n_samples=3000,
        xlim=(-6, 6),
        ylim=(-6, 6)
    )
    
    # Plot marginals
    print("\nPlotting marginal distributions...")
    plot_marginals(model, target, n_samples=2000)
    
    # ==================================================================
    # 6. Compare Moments
    # ==================================================================
    print("\n" + "-"*70)
    print("6. Comparing moments")
    print("-"*70)
    
    print_comparison(model, target, n_samples=5000)



    z_model, _ = model.sample(5000)
    z_model_np = z_model.detach().numpy()

    # Check z₁ distribution - should be bimodal around -2 and +2
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(z_model_np[:, 0], bins=50, alpha=0.7, label='SIVI')
    plt.axvline(-2, color='red', linestyle='--', label='True modes')
    plt.axvline(2, color='red', linestyle='--')
    plt.xlabel('z₁')
    plt.title('z₁ distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(z_model_np[:, 0], z_model_np[:, 1], alpha=0.3, s=5)
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title('2D samples')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()