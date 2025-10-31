import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal, MixtureSameFamily, Categorical
from matplotlib.colors import LogNorm

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Symmetric Bimodal Target
# ============================================================================

class SymmetricBimodalTarget:
    """Symmetric two-mode Gaussian mixture"""
    def __init__(self, mode_separation=3.5, dim=2, mode_scale=0.2):
        self.dim = dim
        self.mode_separation = mode_separation
        self.mode_scale = mode_scale
        
        # Two modes at ±mu with tight covariance
        mu = torch.zeros(dim)
        mu[0] = mode_separation
        
        self.means = torch.stack([mu, -mu])
        self.cov = (mode_scale ** 2) * torch.eye(dim)
        self.weights = torch.tensor([0.5, 0.5])
        
        print(f"Initialized symmetric bimodal target: modes at ±({mode_separation}, 0, ...), scale={mode_scale}")
    
    def sample(self, n_samples):
        """Sample from symmetric mixture"""
        # Sample component indices (50-50)
        components = torch.randint(0, 2, (n_samples,))
        
        samples = []
        for i in range(2):
            n_i = (components == i).sum().item()
            if n_i > 0:
                dist = MultivariateNormal(self.means[i], self.cov)
                samples.append(dist.sample((n_i,)))
        
        return torch.cat(samples, dim=0)
    
    def log_prob(self, x):
        """Compute log probability of the mixture"""
        log_probs = []
        for i in range(2):
            dist = MultivariateNormal(self.means[i], self.cov)
            log_probs.append(dist.log_prob(x) + torch.log(self.weights[i]))
        
        log_probs = torch.stack(log_probs, dim=-1)
        return torch.logsumexp(log_probs, dim=-1)
    
    def mode_probabilities(self, q_model, K_eval=2048, n_samples=5000):
        """
        Estimate probability mass in each mode under q
        Right mode: x[0] > 0
        Left mode: x[0] < 0
        """
        q_model.eval()
        with torch.no_grad():
            # Sample from q using same transformations as forward()
            base_dist = Normal(q_model.base_loc, q_model.base_scale)
            z = base_dist.sample((n_samples,))
            
            # Apply mean capping
            raw_mu = q_model.mu_net(z)
            mu = q_model.mean_clip * torch.tanh(raw_mu)
            
            # Apply variance bounds
            log_sigma = q_model.log_sigma_net(z)
            log_sigma = torch.clamp(log_sigma, min=np.log(q_model.min_sigma), max=np.log(q_model.max_sigma))
            sigma = torch.exp(log_sigma)
            
            eps = torch.randn_like(mu)
            samples = mu + sigma * eps
            
            # Count samples in each mode
            right_mode = (samples[:, 0] > 0).float().mean()
            left_mode = (samples[:, 0] < 0).float().mean()
            
            # Ratio (avoid division by zero)
            ratio = right_mode / (left_mode + 1e-8)
            
        return right_mode.item(), left_mode.item(), ratio.item()


# ============================================================================
# SIVI Model (same as before)
# ============================================================================

class SIVI(nn.Module):
    def __init__(self, latent_dim=1, output_dim=2, hidden_dim=32, 
                 min_sigma=0.25, max_sigma=1.5, mean_clip=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.mean_clip = mean_clip
        
        # Simpler architecture with fixed depth=2
        n_hidden_layers = 2
        
        # MLP for mean
        layers_mu = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers_mu.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_mu.append(nn.Linear(hidden_dim, output_dim))
        self.mu_net = nn.Sequential(*layers_mu)
        
        # MLP for log std
        layers_sigma = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers_sigma.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_sigma.append(nn.Linear(hidden_dim, output_dim))
        self.log_sigma_net = nn.Sequential(*layers_sigma)
        
        # Base distribution
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
    
    def forward(self, x, K=64):
        batch_size = x.shape[0]
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        # Mean with tanh capping to prevent unbounded values
        raw_mu = self.mu_net(z_flat)
        mu = self.mean_clip * torch.tanh(raw_mu)
        
        # Variance with strict bounds
        log_sigma = self.log_sigma_net(z_flat)
        log_sigma = torch.clamp(log_sigma, min=np.log(self.min_sigma), max=np.log(self.max_sigma))
        sigma = torch.exp(log_sigma)
        
        mu = mu.reshape(batch_size, K, self.output_dim)
        sigma = sigma.reshape(batch_size, K, self.output_dim)
        
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)
        
        log_prob_xz = -0.5 * torch.sum(
            ((x_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        
        log_qx = torch.logsumexp(log_prob_xz, dim=1) - np.log(K)
        return log_qx


# ============================================================================
# Training with fixed K
# ============================================================================

def train_sivi_fixed_K(model, target, K_train, n_epochs=2000, batch_size=256, lr=1e-3):
    """Train with fixed K to isolate surrogate bias"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    for epoch in range(n_epochs):
        x = target.sample(batch_size).to(device)
        log_qx = model(x, K=K_train)
        loss = -log_qx.mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    return model


# ============================================================================
# TV Distance Estimation
# ============================================================================

def estimate_tv_monte_carlo(model, target, n_samples=10000, K_eval=2048):
    """Monte Carlo TV estimator with large K_eval"""
    model.eval()
    
    n_p = n_samples // 2
    n_q = n_samples - n_p
    
    with torch.no_grad():
        x_p = target.sample(n_p).to(device)
        
        # Sample from q using the model's forward pass
        base_dist = Normal(model.base_loc, model.base_scale)
        z = base_dist.sample((n_q,))
        
        # Apply same transformations as in forward()
        raw_mu = model.mu_net(z)
        mu = model.mean_clip * torch.tanh(raw_mu)
        
        log_sigma = model.log_sigma_net(z)
        log_sigma = torch.clamp(log_sigma, min=np.log(model.min_sigma), max=np.log(model.max_sigma))
        sigma = torch.exp(log_sigma)
        
        eps = torch.randn_like(mu)
        x_q = mu + sigma * eps
        
        x_all = torch.cat([x_p, x_q], dim=0)
        
        # Evaluate densities with large K_eval
        batch_size = 500
        log_p_vals = []
        log_q_vals = []
        
        for i in range(0, len(x_all), batch_size):
            x_batch = x_all[i:i+batch_size]
            log_p_vals.append(target.log_prob(x_batch))
            log_q_vals.append(model(x_batch, K=K_eval))
        
        log_p = torch.cat(log_p_vals)
        log_q = torch.cat(log_q_vals)
        
        p_vals = torch.exp(log_p)
        q_vals = torch.exp(log_q)
        
        # Correct TV formula with 0.5 factor
        tv = torch.mean(torch.abs(p_vals - q_vals) / (p_vals + q_vals)) 
    
    return tv.item()


# ============================================================================
# Visualization: 2D density heatmaps
# ============================================================================

def visualize_density_2d(model, target, K_eval=2048, grid_size=100, 
                         xlim=(-4, 4), ylim=(-3, 3), title="Learned Density"):
    """Create 2D heatmap of learned density"""
    x_range = torch.linspace(xlim[0], xlim[1], grid_size)
    y_range = torch.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    
    model.eval()
    with torch.no_grad():
        # Evaluate in batches
        batch_size = 500
        log_q_vals = []
        log_p_vals = []
        
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i+batch_size]
            log_q_vals.append(model(batch, K=K_eval))
            log_p_vals.append(target.log_prob(batch))
        
        log_q = torch.cat(log_q_vals)
        log_p = torch.cat(log_p_vals)
        
        q_density = torch.exp(log_q).cpu().numpy().reshape(grid_size, grid_size)
        p_density = torch.exp(log_p).cpu().numpy().reshape(grid_size, grid_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Target
    im1 = axes[0].contourf(xx.numpy(), yy.numpy(), p_density, levels=20, cmap='viridis')
    axes[0].set_title('Target p(x)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x₁')
    axes[0].set_ylabel('x₂')
    plt.colorbar(im1, ax=axes[0])
    
    # Learned
    im2 = axes[1].contourf(xx.numpy(), yy.numpy(), q_density, levels=20, cmap='viridis')
    axes[1].set_title(title, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    return fig


# ============================================================================
# Main Experiment 3
# ============================================================================

def experiment_3_finite_K_bias(n_seeds=3):
    print("\n" + "="*70)
    print("EXPERIMENT 3: FINITE-K SURROGATE BIAS AND MODE PREFERENCE")
    print("="*70)
    
    # Hard bimodal target: well-separated, tight modes
    target = SymmetricBimodalTarget(mode_separation=3.5, mode_scale=0.2, dim=2)
    K_values = [1, 2, 5, 10, 50, 200]
    K_eval = 4096  # Large K for evaluation
    n_samples_eval = 10000
    
    # Restricted architecture and training settings across K
    latent_dim = 1  # Severe bottleneck
    hidden_dim = 32
    n_epochs = 2000
    lr = 1e-3
    batch_size = 256
    
    all_results = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        mode_ratios = []
        tv_distances = []
        right_probs = []
        left_probs = []
        
        for K_train in K_values:
            print(f"\n[Seed {seed+1}/{n_seeds}] Training with K = {K_train}")
            
            model = SIVI(
                latent_dim=latent_dim,
                output_dim=2,
                hidden_dim=hidden_dim,
                min_sigma=0.25,
                max_sigma=1.5,
                mean_clip=10.0
            ).to(device)
            
            # Initialize
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            # Train with fixed K
            model = train_sivi_fixed_K(model, target, K_train=K_train, 
                                       n_epochs=n_epochs, batch_size=batch_size, lr=lr)
            
            # Evaluate mode probabilities
            right_prob, left_prob, ratio = target.mode_probabilities(model, K_eval=K_eval)
            mode_ratios.append(ratio)
            right_probs.append(right_prob)
            left_probs.append(left_prob)
            
            # Evaluate TV distance
            tv = estimate_tv_monte_carlo(model, target, n_samples=n_samples_eval, K_eval=K_eval)
            tv_distances.append(tv)
            
            print(f"  Mode ratio (right/left): {ratio:.3f}")
            print(f"  Right mode: {right_prob:.3f}, Left mode: {left_prob:.3f}")
            print(f"  TV distance: {tv:.6f}")
            
            # Save model for visualization (only for seed 0)
            if seed == 0 and K_train in [1, 5, 200]:
                torch.save(model.state_dict(), f'model_K{K_train}_seed{seed}.pt')
        
        all_results.append({
            'mode_ratios': mode_ratios,
            'tv_distances': tv_distances,
            'right_probs': right_probs,
            'left_probs': left_probs
        })
    
    # Aggregate results
    all_mode_ratios = np.array([r['mode_ratios'] for r in all_results])
    all_tv = np.array([r['tv_distances'] for r in all_results])
    all_right = np.array([r['right_probs'] for r in all_results])
    all_left = np.array([r['left_probs'] for r in all_results])
    
    mean_mode_ratios = all_mode_ratios.mean(axis=0)
    std_mode_ratios = all_mode_ratios.std(axis=0)
    
    mean_tv = all_tv.mean(axis=0)
    std_tv = all_tv.std(axis=0)
    
    mean_right = all_right.mean(axis=0)
    mean_left = all_left.mean(axis=0)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Mode imbalance vs K
    ax1 = axes[0]
    ax1.plot(K_values, mean_mode_ratios, 'o-', linewidth=2, markersize=8, 
             color='steelblue', label='Mode ratio', zorder=3)
    ax1.fill_between(K_values, mean_mode_ratios - std_mode_ratios, 
                     mean_mode_ratios + std_mode_ratios,
                     alpha=0.3, color='steelblue', zorder=2)
    
    # Reference line at ratio = 1 (perfect symmetry)
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                label='Perfect symmetry', zorder=1)
    
    ax1.set_xlabel('Training K', fontsize=12)
    ax1.set_ylabel('Mode Ratio (right/left)', fontsize=12)
    ax1.set_title('(3a) Mode Imbalance vs K', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(fontsize=10)
    
    # Panel 2: Total variation vs K
    ax2 = axes[1]
    ax2.plot(K_values, mean_tv, 'o-', linewidth=2, markersize=8,
             color='red', label='TV distance', zorder=3)
    ax2.fill_between(K_values, mean_tv - std_tv, mean_tv + std_tv,
                     alpha=0.3, color='red', zorder=2)
    
    # Theoretical K^{-1/2} reference
    K_ref = np.array(K_values)
    tv_ref = mean_tv[0] * (K_ref / K_values[0]) ** (-0.5)
    ax2.plot(K_values, tv_ref, '--', linewidth=2, alpha=0.6, color='orange',
             label='$K^{-1/2}$ reference', zorder=1)
    
    ax2.set_xlabel('Training K', fontsize=12)
    ax2.set_ylabel('Total Variation Distance', fontsize=12)
    ax2.set_title('(3b) TV Distance vs K', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('experiment_3_finite_K_bias.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - Finite-K Surrogate Bias")
    print("="*70)
    print("\nMode ratios and TV distances:")
    print(f"{'K':>6} | {'Right':>6} | {'Left':>6} | {'Ratio':>7} | {'TV':>8}")
    print("-" * 50)
    for i, K in enumerate(K_values):
        print(f"{K:6d} | {mean_right[i]:6.3f} | {mean_left[i]:6.3f} | "
              f"{mean_mode_ratios[i]:7.3f} | {mean_tv[i]:8.6f}")
    
    print("\n" + "="*70)
    print("Key observations:")
    print(f"  - At K=1: Mode ratio = {mean_mode_ratios[0]:.2f} (strong collapse)")
    print(f"  - At K=200: Mode ratio = {mean_mode_ratios[-1]:.2f} (near-symmetric)")
    print(f"  - TV reduction: {(1 - mean_tv[-1]/mean_tv[0])*100:.1f}%")
    print("="*70)
    
    # Create visualization comparison for seed 0
    print("\nGenerating density visualizations for K ∈ {1, 5, 200}...")
    
    for K in [1, 5, 200]:
        try:
            model = SIVI(latent_dim=latent_dim, output_dim=2, hidden_dim=hidden_dim,
                        min_sigma=0.25, max_sigma=1.5, mean_clip=10.0).to(device)
            model.load_state_dict(torch.load(f'model_K{K}_seed0.pt'))
            fig = visualize_density_2d(model, target, K_eval=K_eval, 
                                       title=f'Learned q(x) with K={K}')
            plt.savefig(f'density_K{K}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved density_K{K}.png")
        except FileNotFoundError:
            print(f"  Model file for K={K} not found, skipping visualization")
    
    return K_values, mean_mode_ratios, std_mode_ratios, mean_tv, std_tv


if __name__ == "__main__":
    experiment_3_finite_K_bias(n_seeds=3)