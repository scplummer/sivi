"""
Utility functions for SIVI experiments.

Includes:
- Visualization tools
- Metrics computation
- Helper functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def plot_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Plot training history (ELBO over iterations).
    
    Args:
        history: Dictionary with 'elbo' and 'iteration' keys
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.plot(history['iteration'], history['elbo'], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ELBO', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def plot_2d_comparison(
    model,
    target,
    n_samples: int = 1000,
    xlim: Tuple[float, float] = (-6, 6),
    ylim: Tuple[float, float] = (-6, 6),
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot 2D comparison between target and learned distribution.
    
    Shows three panels:
    1. Target distribution samples
    2. SIVI approximation samples  
    3. Overlay of both
    
    Args:
        model: Trained SIVIModel
        target: Target distribution
        n_samples: Number of samples to plot
        xlim: X-axis limits
        ylim: Y-axis limits
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    model.eval()
    
    # Sample from target
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
    
    # Sample from model
    with torch.no_grad():
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Target samples
    axes[0].scatter(z_target[:, 0], z_target[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_xlabel('z₁', fontsize=12)
    axes[0].set_ylabel('z₂', fontsize=12)
    axes[0].set_title('Target Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Model samples
    axes[1].scatter(z_model[:, 0], z_model[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_xlabel('z₁', fontsize=12)
    axes[1].set_ylabel('z₂', fontsize=12)
    axes[1].set_title('SIVI Approximation', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Overlay
    axes[2].scatter(z_target[:, 0], z_target[:, 1], alpha=0.3, s=10, c='blue', label='Target')
    axes[2].scatter(z_model[:, 0], z_model[:, 1], alpha=0.3, s=10, c='red', label='SIVI')
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    axes[2].set_xlabel('z₁', fontsize=12)
    axes[2].set_ylabel('z₂', fontsize=12)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()
    
    model.train()


def plot_2d_density_comparison(
    model,
    target,
    n_samples: int = 5000,
    n_bins: int = 50,
    xlim: Tuple[float, float] = (-6, 6),
    ylim: Tuple[float, float] = (-6, 6),
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot 2D density comparison using hexbin plots.
    
    Args:
        model: Trained SIVIModel
        target: Target distribution
        n_samples: Number of samples for density estimation
        n_bins: Number of bins for hexbin
        xlim: X-axis limits
        ylim: Y-axis limits
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    model.eval()
    
    # Sample from target
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
    
    # Sample from model
    with torch.no_grad():
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Target density
    hb1 = axes[0].hexbin(z_target[:, 0], z_target[:, 1], gridsize=n_bins, cmap='Blues', mincnt=1)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_xlabel('z₁', fontsize=12)
    axes[0].set_ylabel('z₂', fontsize=12)
    axes[0].set_title('Target Density', fontsize=14)
    plt.colorbar(hb1, ax=axes[0], label='Count')
    
    # Panel 2: Model density
    hb2 = axes[1].hexbin(z_model[:, 0], z_model[:, 1], gridsize=n_bins, cmap='Reds', mincnt=1)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_xlabel('z₁', fontsize=12)
    axes[1].set_ylabel('z₂', fontsize=12)
    axes[1].set_title('SIVI Density', fontsize=14)
    plt.colorbar(hb2, ax=axes[1], label='Count')
    
    # Panel 3: Difference (KDE contours)
    from scipy.stats import gaussian_kde
    
    # Compute KDE for both
    kde_target = gaussian_kde(z_target.T)
    kde_model = gaussian_kde(z_model.T)
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate densities
    Z_target = kde_target(positions).reshape(X.shape)
    Z_model = kde_model(positions).reshape(X.shape)
    
    # Plot contours
    axes[2].contour(X, Y, Z_target, levels=5, colors='blue', alpha=0.6, linewidths=2)
    axes[2].contour(X, Y, Z_model, levels=5, colors='red', alpha=0.6, linewidths=2)
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    axes[2].set_xlabel('z₁', fontsize=12)
    axes[2].set_ylabel('z₂', fontsize=12)
    axes[2].set_title('Density Contours (Blue=Target, Red=SIVI)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved density comparison to {save_path}")
    
    plt.show()
    
    model.train()


def plot_marginals(
    model,
    target,
    n_samples: int = 5000,
    n_dims: Optional[int] = None,
    n_bins: int = 50,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
):
    """
    Plot marginal distributions for each dimension.
    
    Args:
        model: Trained SIVIModel
        target: Target distribution
        n_samples: Number of samples
        n_dims: Number of dimensions to plot (None = all)
        n_bins: Number of histogram bins
        figsize: Figure size (auto-computed if None)
        save_path: If provided, save figure to this path
    """
    model.eval()
    
    # Sample from both distributions
    with torch.no_grad():
        z_target = target.sample(n_samples).numpy()
        z_model, _ = model.sample(n_samples)
        z_model = z_model.numpy()
    
    dim = z_target.shape[1]
    n_dims = n_dims or dim
    n_dims = min(n_dims, dim)
    
    # Auto-compute figure size
    if figsize is None:
        n_cols = min(n_dims, 4)
        n_rows = int(np.ceil(n_dims / n_cols))
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(int(np.ceil(n_dims / 4)), min(n_dims, 4), figsize=figsize)
    if n_dims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # TODO: Resolve Pylance conflict. 
    for i in range(n_dims):
        axes[i].hist(z_target[:, i], bins=n_bins, alpha=0.5, label='Target', 
                    density=True, color='blue', edgecolor='black')
        axes[i].hist(z_model[:, i], bins=n_bins, alpha=0.5, label='SIVI',
                    density=True, color='red', edgecolor='black')
        axes[i].set_xlabel(f'z_{i+1}', fontsize=10)
        axes[i].set_ylabel('Density', fontsize=10)
        axes[i].set_title(f'Dimension {i+1}', fontsize=12)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved marginal plots to {save_path}")
    
    plt.show()
    
    model.train()


def compute_moments(samples: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute first and second moments of samples.
    
    Args:
        samples: Tensor of shape (n_samples, dim)
        
    Returns:
        Dictionary with 'mean', 'std', 'cov' keys
    """
    return {
        'mean': samples.mean(dim=0),
        'std': samples.std(dim=0),
        'cov': torch.cov(samples.T)
    }


def compare_moments(model, target, n_samples: int = 5000) -> Dict:
    """
    Compare moments between target and model.
    
    Args:
        model: Trained SIVIModel
        target: Target distribution
        n_samples: Number of samples
        
    Returns:
        Dictionary with target and model moments, plus errors
    """
    model.eval()
    
    with torch.no_grad():
        z_target = target.sample(n_samples)
        z_model, _ = model.sample(n_samples)
    
    moments_target = compute_moments(z_target)
    moments_model = compute_moments(z_model)
    
    results = {
        'target': moments_target,
        'model': moments_model,
        'mean_error': torch.abs(moments_target['mean'] - moments_model['mean']),
        'std_error': torch.abs(moments_target['std'] - moments_model['std'])
    }
    
    model.train()
    return results


def print_comparison(model, target, n_samples: int = 5000):
    """
    Print a formatted comparison of moments.
    
    Args:
        model: Trained SIVIModel
        target: Target distribution
        n_samples: Number of samples
    """
    results = compare_moments(model, target, n_samples)
    
    print("\n" + "="*60)
    print("MOMENT COMPARISON")
    print("="*60)
    
    print("\nMean:")
    print(f"  Target: {results['target']['mean'].numpy()}")
    print(f"  Model:  {results['model']['mean'].numpy()}")
    print(f"  Error:  {results['mean_error'].numpy()}")
    
    print("\nStandard Deviation:")
    print(f"  Target: {results['target']['std'].numpy()}")
    print(f"  Model:  {results['model']['std'].numpy()}")
    print(f"  Error:  {results['std_error'].numpy()}")
    
    print("\nCovariance (Target):")
    print(results['target']['cov'].numpy())
    print("\nCovariance (Model):")
    print(results['model']['cov'].numpy())
    print("="*60 + "\n")


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_utils():
    """Test utility functions."""
    print("Testing utils...\n")
    
    from model import SIVIModel
    from target import GaussianMixture
    from trainer import train
    
    # Set seed
    set_random_seed(42)
    print(" Random seed set")
    
    # Create and train model
    target = GaussianMixture.create_2d_bimodal(separation=4.0)
    model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
    
    history = train(model, target, n_iterations=100, n_samples=50, K=3, 
                   learning_rate=1e-2, print_every=100)
    print(" Model trained")
    
    # Test moment comparison
    print("\nTesting moment comparison...")
    print_comparison(model, target, n_samples=1000)
    print(" Moment comparison works")
    
    print("\nAll utility tests passed!")
    print("(Skipping plots in test mode)")


if __name__ == "__main__":
    test_utils()
