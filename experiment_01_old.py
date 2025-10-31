import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Target Distribution: 3-component Gaussian mixture on [-1,1]^2
# ============================================================================

class TargetMixture:
    def __init__(self):
        # 3 Gaussian components with means spread out in [-1,1]^2
        self.means = torch.tensor([
            [-0.5, -0.5],
            [0.5, 0.5],
            [0.0, 0.6]
        ], dtype=torch.float32)
        
        self.covs = torch.stack([
            torch.tensor([[0.05, 0.01], [0.01, 0.05]], dtype=torch.float32),
            torch.tensor([[0.04, -0.01], [-0.01, 0.06]], dtype=torch.float32),
            torch.tensor([[0.03, 0.0], [0.0, 0.03]], dtype=torch.float32)
        ])
        
        self.weights = torch.tensor([0.3, 0.4, 0.3])
        
    def sample(self, n_samples):
        """Sample from the mixture"""
        # Sample component indices
        components = torch.multinomial(self.weights, n_samples, replacement=True)
        
        samples = []
        for i in range(3):
            n_i = (components == i).sum().item()
            if n_i > 0:
                dist = MultivariateNormal(self.means[i], self.covs[i])
                samples.append(dist.sample((n_i,)))
        
        samples = torch.cat(samples, dim=0)
        
        # Truncate to [-1, 1]^2
        samples = torch.clamp(samples, -1.0, 1.0)
        
        return samples
    
    def log_prob(self, x):
        """Compute log probability of the mixture"""
        log_probs = []
        for i in range(3):
            dist = MultivariateNormal(self.means[i], self.covs[i])
            log_probs.append(dist.log_prob(x) + torch.log(self.weights[i]))
        
        log_probs = torch.stack(log_probs, dim=-1)
        return torch.logsumexp(log_probs, dim=-1)


# ============================================================================
# SIVI Model with Gaussian kernel
# ============================================================================

class SIVI(nn.Module):
    def __init__(self, latent_dim=2, output_dim=2, hidden_dim=64, depth=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # MLP for mean
        layers_mu = []
        layers_mu.append(nn.Linear(latent_dim, hidden_dim))
        layers_mu.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers_mu.append(nn.Linear(hidden_dim, hidden_dim))
            layers_mu.append(nn.ReLU())
        
        layers_mu.append(nn.Linear(hidden_dim, output_dim))
        self.mu_net = nn.Sequential(*layers_mu)
        
        # MLP for log std (diagonal covariance)
        layers_sigma = []
        layers_sigma.append(nn.Linear(latent_dim, hidden_dim))
        layers_sigma.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers_sigma.append(nn.Linear(hidden_dim, hidden_dim))
            layers_sigma.append(nn.ReLU())
        
        layers_sigma.append(nn.Linear(hidden_dim, output_dim))
        self.log_sigma_net = nn.Sequential(*layers_sigma)
        
        # Base distribution for z
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
    
    def forward(self, x, K=64):
        """
        Compute log q(x) using K samples from the base distribution
        x: (batch_size, output_dim)
        Returns: (batch_size,)
        """
        batch_size = x.shape[0]
        
        # Sample K latent variables for each x
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))  # (batch_size, K, latent_dim)
        
        # Reshape for batch processing
        z_flat = z.reshape(-1, self.latent_dim)  # (batch_size * K, latent_dim)
        
        # Get parameters
        mu = self.mu_net(z_flat)  # (batch_size * K, output_dim)
        log_sigma = self.log_sigma_net(z_flat)  # (batch_size * K, output_dim)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        
        # Reshape back
        mu = mu.reshape(batch_size, K, self.output_dim)
        sigma = sigma.reshape(batch_size, K, self.output_dim)
        
        # Compute log p(x|z) for each z
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)  # (batch_size, K, output_dim)
        
        # Log probability under diagonal Gaussian
        log_prob_xz = -0.5 * torch.sum(
            ((x_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )  # (batch_size, K)
        
        # Log q(x) = log mean_k exp(log p(x|z_k))
        log_qx = torch.logsumexp(log_prob_xz, dim=1) - np.log(K)
        
        return log_qx
    
    def sample(self, n_samples, K=1):
        """Sample from the SIVI distribution"""
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((n_samples * K,))
        
        mu = self.mu_net(z)
        log_sigma = self.log_sigma_net(z)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=10.0)
        
        eps = torch.randn_like(mu)
        samples = mu + sigma * eps
        
        return samples


# ============================================================================
# Training function
# ============================================================================

def train_sivi(model, target, n_epochs=2000, batch_size=256, lr=1e-3, K=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    for epoch in range(n_epochs):
        x = target.sample(batch_size).to(device)
        log_qx = model(x, K=K)
        loss = -log_qx.mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model

# ============================================================================
# TV distance estimation via Monte Carlo
# ============================================================================

def estimate_tv_distance(model, target, n_samples=10000, grid_size=50, K=64):
    """
    Estimate TV distance using Monte Carlo integration on a grid
    TV = 0.5 * integral |p(x) - q(x)| dx
    """
    # Create grid over [-1, 1]^2
    x_range = torch.linspace(-1, 1, grid_size)
    y_range = torch.linspace(-1, 1, grid_size)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    
    # Compute densities on grid
    with torch.no_grad():
        log_q = model(grid_points, K=K)
        q_vals = torch.exp(log_q)
        
        log_p = target.log_prob(grid_points)
        p_vals = torch.exp(log_p)
    
    # Numerical integration (Riemann sum)
    dx = 2.0 / grid_size
    dy = 2.0 / grid_size
    area_element = dx * dy
    
    tv = 0.5 * torch.sum(torch.abs(p_vals - q_vals)) * area_element
    
    return tv.item()


# ============================================================================
# Main experiment: TV vs Width
# ============================================================================

def run_experiment():
    target = TargetMixture()
    widths = [8, 16, 32, 64, 128, 256]
    n_seeds = 3  # Run multiple times
    K = 128

    all_results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        tv_distances = []
        for width in widths:
            print(f"\n[Seed {seed+1}/{n_seeds}] Width = {width}")
            model = SIVI(latent_dim=2, output_dim=2, hidden_dim=width, depth=2).to(device)
            
            # Better initialization
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            model = train_sivi(model, target, n_epochs=2000, batch_size=256, lr=5e-4, K=K)
            model.eval()
            tv = estimate_tv_distance(model, target, grid_size=50, K=K)
            tv_distances.append(tv)
            print(f"TV: {tv:.6f}")
        
        all_results.append(tv_distances)
    
    # Average and std
    all_results = np.array(all_results)
    mean_tv = all_results.mean(axis=0)
    std_tv = all_results.std(axis=0)
    
    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(widths, mean_tv, yerr=std_tv, fmt='o-', linewidth=2, 
                 markersize=8, capsize=5, label='SIVI (mean ± std)')
    plt.xlabel('Network Width (W)', fontsize=14)
    plt.ylabel('Total Variation Distance', fontsize=14)
    plt.title('Compact Approximation: TV vs Network Width', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    beta_m = 0.25
    theoretical = mean_tv[0] * (np.array(widths) / widths[0]) ** (-beta_m)
    plt.plot(widths, theoretical, '--', alpha=0.5, color='orange', 
             label=f'$W^{{-{beta_m}}}$ (theory)')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('tv_vs_width.png', dpi=300)
    plt.show()
    
    print("\n" + "="*60)
    print("RESULTS (averaged over {} seeds)".format(n_seeds))
    print("="*60)
    for w, m, s in zip(widths, mean_tv, std_tv):
        print(f"Width: {w:4d}  ->  TV: {m:.6f} ± {s:.6f}")
    
    return widths, mean_tv, std_tv

if __name__ == "__main__":
    widths, mean_tv, std_tv = run_experiment()