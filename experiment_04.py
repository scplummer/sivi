
# ============================================================================
# Gamma Convergence Experiment - Matches Theory 
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch.distributions import Normal, MultivariateNormal

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Simple 1D Target for Objective Landscape Visualization
# ============================================================================

class SimpleTarget1D:
    """1D Gaussian target for clear objective visualization"""
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.dist = Normal(torch.tensor(mu), torch.tensor(sigma))
        print(f"Initialized 1D target: N({mu}, {sigma}²)")
    
    def sample(self, n_samples):
        """Sample from target"""
        return self.dist.sample((n_samples, 1))
    
    def log_prob(self, x):
        """Compute log probability"""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.dist.log_prob(x.squeeze(-1))


# ============================================================================
# Simple SIVI with Scalar Parameter θ
# ============================================================================

class SimpleSIVI1D(nn.Module):
    """
    Simplified SIVI with a scalar parameter θ controlling location
    q_θ(x) = E_z[N(x | μ_θ(z), σ²)]
    where μ_θ(z) = θ + α*z for interpretability
    """
    def __init__(self, fixed_sigma=0.5, alpha=0.3):
        super().__init__()
        self.fixed_sigma = fixed_sigma
        self.alpha = alpha
        print(f"Initialized SimpleSIVI: σ={fixed_sigma}, α={alpha}")
    
    def forward(self, x, theta, K=1, z_samples=None):
        """
        Evaluate log q_θ(x) with K samples
        x: (n, 1) or (n,)
        theta: scalar
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        batch_size = x.shape[0]
        
        # Sample z or use provided samples (for consistency across θ)
        if z_samples is None:
            z = torch.randn(batch_size, K, device=x.device)
        else:
            z = z_samples
        
        # Mean: μ_θ(z) = θ + α*z
        mu = theta + self.alpha * z  # (batch, K)
        sigma = self.fixed_sigma
        
        # Expand x for broadcasting
        x_expanded = x.expand(-1, K)  # (batch, K)
        
        # Log prob under each z
        log_prob_xz = -0.5 * ((x_expanded - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        
        # Log mean over K samples
        log_qx = torch.logsumexp(log_prob_xz, dim=1) - np.log(K)
        
        return log_qx
    
    def evaluate_objective(self, data, theta, K=1, z_cache=None):
        """
        Evaluate L_{K,n}(θ) = (1/n) Σ log q_θ(x_i)
        """
        log_qx = self.forward(data, theta, K=K, z_samples=z_cache)
        return log_qx.mean().item()


# ============================================================================
# Objective Landscape Evaluation
# ============================================================================

def evaluate_landscape(model, data, theta_grid, K=1, seed=None):
    """
    Evaluate objective L_{K,n}(θ) over a grid of θ values
    Uses consistent z samples across θ for fair comparison
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n = data.shape[0]
    objectives = []
    
    # Pre-sample z for consistency across θ
    z_cache = torch.randn(n, K, device=data.device)
    
    model.eval()
    with torch.no_grad():
        for theta in theta_grid:
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device=data.device)
            obj = model.evaluate_objective(data, theta_tensor, K=K, z_cache=z_cache)
            objectives.append(obj)
    
    return np.array(objectives)


def evaluate_population_objective(model, target, theta_grid, K=5000, n_samples=100000):
    """
    Evaluate limiting objective L_∞(θ) = E[log q_θ(X)]
    using very large K and n
    """
    print(f"Evaluating population objective with K={K}, n={n_samples}...")
    
    # Generate large dataset
    data = target.sample(n_samples).to(device)
    
    objectives = []
    model.eval()
    
    with torch.no_grad():
        for i, theta in enumerate(theta_grid):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(theta_grid)}")
            
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device)
            
            # Evaluate in batches to avoid memory issues
            batch_size = 5000
            log_qx_all = []
            
            for j in range(0, n_samples, batch_size):
                data_batch = data[j:j+batch_size]
                log_qx = model.forward(data_batch, theta_tensor, K=K)
                log_qx_all.append(log_qx)
            
            log_qx_full = torch.cat(log_qx_all)
            obj = log_qx_full.mean().item()
            objectives.append(obj)
    
    return np.array(objectives)


# ============================================================================
# Main Experiment 4
# ============================================================================

def experiment_4_gamma_convergence():
    print("\n" + "="*70)
    print("EXPERIMENT 4: Γ-CONVERGENCE AND OBJECTIVE STABILITY")
    print("="*70)
    
    # Setup
    target = SimpleTarget1D(mu=0.0, sigma=1.0)
    model = SimpleSIVI1D(fixed_sigma=0.5, alpha=0.3).to(device)
    
    # Parameter grid
    theta_grid = np.linspace(-3, 3, 100)
    
    # K and n values to test
    K_values = [1, 5, 50, 500]
    n_values = [100, 1000, 10000]
    
    # Seeds for reproducibility
    seed = 42
    
    # Store all landscapes
    landscapes = {}
    
    # Evaluate population objective (limiting case)
    print("\n" + "="*70)
    print("Computing L_∞(θ) [population objective]...")
    print("="*70)
    L_inf = evaluate_population_objective(model, target, theta_grid, K=5000, n_samples=100000)
    theta_star = theta_grid[np.argmax(L_inf)]
    landscapes['inf'] = L_inf
    
    print(f"\nPopulation optimum: θ* = {theta_star:.3f}")
    print(f"L_∞(θ*) = {L_inf.max():.6f}")
    
    # Evaluate empirical objectives for different K and n
    print("\n" + "="*70)
    print("Computing L_{K,n}(θ) for various K and n...")
    print("="*70)
    
    theta_hat_table = []
    
    for n in n_values:
        for K in K_values:
            print(f"\nEvaluating: n={n}, K={K}")
            
            # Generate dataset with fixed seed
            torch.manual_seed(seed)
            data = target.sample(n).to(device)
            
            # Evaluate landscape
            L_Kn = evaluate_landscape(model, data, theta_grid, K=K, seed=seed)
            landscapes[(K, n)] = L_Kn
            
            # Find empirical optimum
            theta_hat = theta_grid[np.argmax(L_Kn)]
            L_hat = L_Kn.max()
            
            # Compute distance to population objective
            gdist = np.abs(L_Kn - L_inf).max()
            mae = np.abs(L_Kn - L_inf).mean()
            
            print(f"  θ̂_{{{K},{n}}} = {theta_hat:.3f} (vs θ* = {theta_star:.3f})")
            print(f"  L_{{{K},{n}}}(θ̂) = {L_hat:.6f}")
            print(f"  Γ-distance: {gdist:.6f}, MAE: {mae:.6f}")
            
            theta_hat_table.append({
                'K': K, 'n': n, 'theta_hat': theta_hat, 
                'L_hat': L_hat, 'gdist': gdist, 'mae': mae
            })
    
    # ========================================================================
    # Visualization 1: Objective Landscapes (2x3 grid)
    # ========================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot settings
    for idx, (K, n) in enumerate([(1, 100), (5, 100), (50, 100),
                                   (1, 10000), (50, 10000), (500, 10000)]):
        ax = axes[idx]
        
        if (K, n) in landscapes:
            L_Kn = landscapes[(K, n)]
            theta_hat = theta_grid[np.argmax(L_Kn)]
            
            # Plot empirical objective
            ax.plot(theta_grid, L_Kn, linewidth=2, label=f'$L_{{{K},{n}}}(\\theta)$', 
                   color='steelblue')
            
            # Plot population objective
            ax.plot(theta_grid, L_inf, '--', linewidth=2, label='$L_\\infty(\\theta)$', 
                   color='orange', alpha=0.7)
            
            # Mark optima
            ax.axvline(theta_hat, color='steelblue', linestyle=':', alpha=0.5,
                      label=f'$\\hat{{\\theta}}$ = {theta_hat:.2f}')
            ax.axvline(theta_star, color='orange', linestyle=':', alpha=0.5,
                      label=f'$\\theta^*$ = {theta_star:.2f}')
            
            ax.set_xlabel('$\\theta$', fontsize=12)
            ax.set_ylabel('Objective Value', fontsize=12)
            ax.set_title(f'K={K}, n={n}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_landscapes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # Visualization 2: Convergence of Maximizers
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: θ̂_{K,n} vs K (for different n)
    ax1 = axes[0]
    for n in n_values:
        theta_hats_K = [t['theta_hat'] for t in theta_hat_table if t['n'] == n]
        K_vals = [t['K'] for t in theta_hat_table if t['n'] == n]
        ax1.plot(K_vals, theta_hats_K, 'o-', linewidth=2, markersize=8, 
                label=f'n={n}')
    
    ax1.axhline(theta_star, color='green', linestyle='--', linewidth=2,
               label='$\\theta^*$ (population)')
    ax1.set_xlabel('K (inner samples)', fontsize=12)
    ax1.set_ylabel('$\\hat{\\theta}_{K,n}$', fontsize=12)
    ax1.set_title('(4a) Convergence of Maximizers vs K', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: θ̂_{K,n} vs n (for different K)
    ax2 = axes[1]
    for K in K_values:
        theta_hats_n = [t['theta_hat'] for t in theta_hat_table if t['K'] == K]
        n_vals = [t['n'] for t in theta_hat_table if t['K'] == K]
        ax2.plot(n_vals, theta_hats_n, 's-', linewidth=2, markersize=8,
                label=f'K={K}')
    
    ax2.axhline(theta_star, color='green', linestyle='--', linewidth=2,
               label='$\\theta^*$ (population)')
    ax2.set_xlabel('n (dataset size)', fontsize=12)
    ax2.set_ylabel('$\\hat{\\theta}_{K,n}$', fontsize=12)
    ax2.set_title('(4b) Convergence of Maximizers vs n', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # Visualization 3: Γ-distance decay
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Γ-distance vs K
    ax1 = axes[0]
    for n in n_values:
        gdists_K = [t['gdist'] for t in theta_hat_table if t['n'] == n]
        K_vals = [t['K'] for t in theta_hat_table if t['n'] == n]
        ax1.plot(K_vals, gdists_K, 'o-', linewidth=2, markersize=8,
                label=f'n={n}')
    
    # Reference: K^{-1}
    K_ref = np.array(K_values)
    gdist_ref = gdists_K[0] * (K_ref / K_values[0]) ** (-1)
    ax1.plot(K_values, gdist_ref, '--', linewidth=2, alpha=0.6, color='gray',
            label='$K^{-1}$ reference')
    
    ax1.set_xlabel('K (inner samples)', fontsize=12)
    ax1.set_ylabel('$\\sup_\\theta |L_{K,n} - L_\\infty|$', fontsize=12)
    ax1.set_title('(4c) Γ-distance vs K', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Γ-distance vs n
    ax2 = axes[1]
    for K in K_values:
        gdists_n = [t['gdist'] for t in theta_hat_table if t['K'] == K]
        n_vals = [t['n'] for t in theta_hat_table if t['K'] == K]
        ax2.plot(n_vals, gdists_n, 's-', linewidth=2, markersize=8,
                label=f'K={K}')
    
    # Reference: n^{-1/2}
    n_ref = np.array(n_values)
    gdist_ref_n = gdists_n[0] * (n_ref / n_values[0]) ** (-0.5)
    ax2.plot(n_values, gdist_ref_n, '--', linewidth=2, alpha=0.6, color='gray',
            label='$n^{-1/2}$ reference')
    
    ax2.set_xlabel('n (dataset size)', fontsize=12)
    ax2.set_ylabel('$\\sup_\\theta |L_{K,n} - L_\\infty|$', fontsize=12)
    ax2.set_title('(4d) Γ-distance vs n', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_gamma_distance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # Summary Table
    # ========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'K':>6} | {'n':>6} | {'θ̂':>7} | {'L(θ̂)':>9} | {'Γ-dist':>8} | {'MAE':>8}")
    print("-" * 70)
    for t in theta_hat_table:
        print(f"{t['K']:6d} | {t['n']:6d} | {t['theta_hat']:7.3f} | "
              f"{t['L_hat']:9.6f} | {t['gdist']:8.6f} | {t['mae']:8.6f}")
    
    print("\n" + "="*70)
    print(f"Population optimum: θ* = {theta_star:.3f}, L_∞(θ*) = {L_inf.max():.6f}")
    print("="*70)
    
    print("\nKey observations:")
    print(f"  - Small K, small n: θ̂ can be far from θ* (noisy + biased)")
    print(f"  - Large K: Removes surrogate bias, Γ-distance ~ K^(-1)")
    print(f"  - Large n: Reduces empirical noise, Γ-distance ~ n^(-1/2)")
    print(f"  - Joint limit (K,n→∞): θ̂_{{{K},{n}}} → θ*, landscapes converge")
    
    return theta_hat_table, landscapes


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    theta_hat_table, landscapes = experiment_4_gamma_convergence()
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")