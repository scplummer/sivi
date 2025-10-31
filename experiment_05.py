"""
Experiment 5: Branch-Collapse Lower Bound

Goal: Empirically certify a strict KL lower bound whenever q drops a branch/mode of p.
This demonstrates that mode-dropping leads to a positive KL gap that can be lower-bounded
using simple Bernoulli divergence on branch events.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal
from scipy.stats import chi2

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Target: 3-branch Gaussian mixture with tight, well-separated modes
# ============================================================================

class ThreeBranchMixture:
    """Three tight Gaussians arranged on a circle arc or line"""
    def __init__(self, separation='normal', dimension=2):
        """
        Args:
            separation: 'normal' (well-separated), 'collision' (two branches close)
        """
        self.dimension = dimension
        
        if separation == 'normal':
            # Well-separated branches on a circle arc
            self.centers = torch.tensor([
                [-0.6, -0.6],
                [0.6, -0.6],
                [0.0, 0.7]
            ], dtype=torch.float32)
        else:  # collision
            # Two branches close together (harder case)
            self.centers = torch.tensor([
                [-0.4, -0.5],
                [-0.2, -0.5],  # Close to first!
                [0.5, 0.6]
            ], dtype=torch.float32)
        
        # Tight covariance (0.15^2 * I)
        cov_val = 0.15**2
        self.covs = torch.stack([
            torch.eye(dimension) * cov_val for _ in range(3)
        ])
        
        # Equal weights
        self.weights = torch.tensor([1/3, 1/3, 1/3])
        
    def sample(self, n_samples):
        """Sample from the mixture"""
        components = torch.multinomial(self.weights, n_samples, replacement=True)
        
        samples = []
        for i in range(3):
            n_i = (components == i).sum().item()
            if n_i > 0:
                dist = MultivariateNormal(self.centers[i], self.covs[i])
                samples.append(dist.sample((n_i,)))
        
        return torch.cat(samples, dim=0)
    
    def log_prob(self, x):
        """Compute log probability"""
        log_probs = []
        for i in range(3):
            dist = MultivariateNormal(self.centers[i], self.covs[i])
            log_probs.append(dist.log_prob(x) + torch.log(self.weights[i]))
        
        log_probs = torch.stack(log_probs, dim=-1)
        return torch.logsumexp(log_probs, dim=-1)


# ============================================================================
# Restricted SIVI: cannot represent 3 separated lobes
# ============================================================================

class RestrictedSIVI(nn.Module):
    """SIVI with latent_dim=1 and small width - cannot capture 3 modes"""
    def __init__(self, latent_dim=1, output_dim=2, hidden_dim=16, depth=2):
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
        
        # MLP for log std (restricted range)
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
        """Compute log q(x) using K samples"""
        batch_size = x.shape[0]
        
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        mu = self.mu_net(z_flat)
        log_sigma = self.log_sigma_net(z_flat)
        # Restrict sigma to [0.2, 1.5] to prevent mode splitting
        sigma = torch.exp(log_sigma).clamp(min=0.2, max=1.5)
        
        mu = mu.reshape(batch_size, K, self.output_dim)
        sigma = sigma.reshape(batch_size, K, self.output_dim)
        
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)
        
        log_prob_xz = -0.5 * torch.sum(
            ((x_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        
        log_qx = torch.logsumexp(log_prob_xz, dim=1) - np.log(K)
        
        return log_qx
    
    def sample(self, n_samples, K=1):
        """Sample from the SIVI distribution"""
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((n_samples * K,))
        
        mu = self.mu_net(z)
        log_sigma = self.log_sigma_net(z)
        sigma = torch.exp(log_sigma).clamp(min=0.2, max=1.5)
        
        eps = torch.randn_like(mu)
        samples = mu + sigma * eps
        
        return samples


# ============================================================================
# Branch event detection and Bernoulli KL bound
# ============================================================================

def branch_events(points, centers, r=0.5):
    """
    Returns binary masks indicating which points fall near each branch
    
    Args:
        points: (N, d) tensor
        centers: (K, d) tensor of branch centers
        r: radius defining "near" the branch
    
    Returns:
        List of (N,) boolean tensors, one per branch
    """
    dists = torch.stack([
        torch.linalg.norm(points - c.unsqueeze(0), dim=1) 
        for c in centers
    ], dim=1)  # (N, K)
    
    return [(dists[:, j] <= r) for j in range(len(centers))]


@torch.no_grad()
def kl_branch_lower_bound(p_sampler, q_sampler, centers, r=0.5, N=20000):
    """
    Compute lower bound on KL(p||q) using Bernoulli divergence on branch events
    
    KL(p||q) >= max_j KL(Bern(p(A_j)) || Bern(q(A_j)))
    
    where A_j = {x : ||x - c_j|| <= r} is the event "near branch j"
    """
    xp = p_sampler(N)
    xq = q_sampler(N)
    
    masks_p = branch_events(xp, centers, r)
    masks_q = branch_events(xq, centers, r)
    
    def bernoulli_kl(p, q):
        """KL divergence between two Bernoulli distributions"""
        p = p.clamp(1e-12, 1 - 1e-12)
        q = q.clamp(1e-12, 1 - 1e-12)
        return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).item()
    
    branch_kls = []
    branch_probs_p = []
    branch_probs_q = []
    
    for mp, mq in zip(masks_p, masks_q):
        prob_p = mp.float().mean()
        prob_q = mq.float().mean()
        kl = bernoulli_kl(prob_p, prob_q)
        
        branch_kls.append(kl)
        branch_probs_p.append(prob_p.item())
        branch_probs_q.append(prob_q.item())
    
    lower_bound = max(branch_kls)
    
    return {
        'lower_bound': lower_bound,
        'branch_kls': branch_kls,
        'branch_probs_p': branch_probs_p,
        'branch_probs_q': branch_probs_q
    }


@torch.no_grad()
def estimate_kl_direct(p, q, N=20000, K_eval=4096):
    """
    Estimate KL(p||q) = E_p[log p(x) - log q(x)] via Monte Carlo
    
    Uses large K_eval to avoid Jensen bias
    """
    x = p.sample(N).to(device)
    
    log_p = p.log_prob(x)
    
    # Compute log_q in batches
    log_q_list = []
    B = 2048
    for i in range(0, N, B):
        log_q_list.append(q(x[i:i+B], K=K_eval))
    log_q = torch.cat(log_q_list)
    
    kl = torch.mean(log_p - log_q)
    se = torch.std(log_p - log_q) / np.sqrt(N)
    
    return kl.item(), se.item()


# ============================================================================
# Training
# ============================================================================

def train_restricted_sivi(model, target, n_epochs=3000, batch_size=256, lr=5e-4, K=128):
    """Train the restricted SIVI model"""
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
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    return model


# ============================================================================
# Visualization
# ============================================================================

def plot_distributions_and_bounds(target, model, r=0.5, save_path='branch_collapse.png'):
    """
    Create comprehensive visualization:
    1. Heatmaps of p and q
    2. Bar chart of branch probabilities
    3. KL and branch-bound comparison
    """
    fig = plt.figure(figsize=(18, 5))
    
    # Create grid for heatmaps
    x_range = np.linspace(-1.2, 1.2, 100)
    y_range = np.linspace(-1.2, 1.2, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=-1), 
                               dtype=torch.float32).to(device)
    
    # Compute densities
    with torch.no_grad():
        log_p = target.log_prob(grid_points)
        p_vals = torch.exp(log_p).cpu().numpy().reshape(100, 100)
        
        log_q = model(grid_points, K=2048)
        q_vals = torch.exp(log_q).cpu().numpy().reshape(100, 100)
    
    # Panel 1: p(x) heatmap
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.contourf(xx, yy, p_vals, levels=20, cmap='Reds')
    # Mark centers
    centers_np = target.centers.numpy()
    ax1.scatter(centers_np[:, 0], centers_np[:, 1], c='black', s=100, 
                marker='x', linewidths=3, label='Branch centers')
    # Draw circles for branch regions
    for c in centers_np:
        circle = plt.Circle(c, r, fill=False, edgecolor='blue', linestyle='--', linewidth=2)
        ax1.add_patch(circle)
    ax1.set_title('Target p(x) - Three Branches', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # Panel 2: q(x) heatmap
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.contourf(xx, yy, q_vals, levels=20, cmap='Blues')
    ax2.scatter(centers_np[:, 0], centers_np[:, 1], c='black', s=100, 
                marker='x', linewidths=3, label='Target centers')
    for c in centers_np:
        circle = plt.Circle(c, r, fill=False, edgecolor='blue', linestyle='--', linewidth=2)
        ax2.add_patch(circle)
    ax2.set_title('Restricted SIVI q(x) - Mode Collapse', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # Panel 3: Branch probabilities and KL bounds
    ax3 = plt.subplot(1, 3, 3)
    
    # Compute branch statistics
    bound_info = kl_branch_lower_bound(
        lambda n: target.sample(n),
        lambda n: model.sample(n),
        target.centers,
        r=r,
        N=20000
    )
    
    kl_direct, kl_se = estimate_kl_direct(target, model, N=20000, K_eval=4096)
    
    # Bar chart of branch probabilities
    x_pos = np.arange(3)
    width = 0.35
    
    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x_pos - width/2, bound_info['branch_probs_p'], width, 
                    label='p(Aⱼ)', color='red', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, bound_info['branch_probs_q'], width, 
                    label='q(Aⱼ)', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Branch Index j', fontsize=12)
    ax3.set_ylabel('Branch Probability', fontsize=12)
    ax3.set_title('Branch Probabilities & KL Bounds', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Branch 1', 'Branch 2', 'Branch 3'])
    ax3.legend(loc='upper left')
    ax3.set_ylim([0, 0.5])
    
    # KL values on right y-axis
    ax3_twin.axhline(y=kl_direct, color='green', linestyle='-', linewidth=3, 
                     label=f'KL(p||q) = {kl_direct:.4f}')
    ax3_twin.axhline(y=bound_info['lower_bound'], color='orange', linestyle='--', 
                     linewidth=3, label=f'Branch Bound = {bound_info["lower_bound"]:.4f}')
    ax3_twin.set_ylabel('KL Divergence', fontsize=12)
    ax3_twin.legend(loc='upper right')
    ax3_twin.set_ylim([0, max(kl_direct * 1.2, 0.5)])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return bound_info, kl_direct, kl_se


# ============================================================================
# Main experiment
# ============================================================================

def run_branch_collapse_experiment():
    """
    Main experiment: Train restricted SIVI on 3-branch target and verify
    that the branch-collapse bound is strictly positive
    """
    print("="*70)
    print("EXPERIMENT 6: BRANCH-COLLAPSE LOWER BOUND")
    print("="*70)
    
    # Create 3-branch target
    print("\n[1/4] Creating 3-branch Gaussian mixture target...")
    target = ThreeBranchMixture(separation='normal', dimension=2)
    print(f"  Branch centers:\n{target.centers}")
    print(f"  Branch weights: {target.weights}")
    
    # Create restricted SIVI (cannot represent 3 modes)
    print("\n[2/4] Creating restricted SIVI (latent_dim=1, width=16)...")
    model = RestrictedSIVI(latent_dim=1, output_dim=2, hidden_dim=16, depth=2).to(device)
    
    # Initialize
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    # Train
    print("\n[3/4] Training restricted SIVI...")
    model = train_restricted_sivi(model, target, n_epochs=3000, batch_size=256, lr=5e-4, K=128)
    model.eval()
    
    # Evaluate and visualize
    print("\n[4/4] Computing branch bounds and visualizing...")
    r = 0.5  # Radius for branch regions (captures >90% of each branch)
    
    bound_info, kl_direct, kl_se = plot_distributions_and_bounds(
        target, model, r=r, save_path='branch_collapse.png'
    )
    
    # Print detailed results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTrue KL(p||q):        {kl_direct:.6f} ± {kl_se:.6f}")
    print(f"Branch Lower Bound:   {bound_info['lower_bound']:.6f}")
    print(f"Gap (should be ≥ 0):  {kl_direct - bound_info['lower_bound']:.6f}")
    
    print(f"\nBranch probabilities (r = {r}):")
    for j in range(3):
        print(f"  Branch {j+1}: p(A{j+1}) = {bound_info['branch_probs_p'][j]:.4f}, "
              f"q(A{j+1}) = {bound_info['branch_probs_q'][j]:.4f}, "
              f"KL_j = {bound_info['branch_kls'][j]:.6f}")
    
    # Acceptance criteria check
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA")
    print("="*70)
    
    # Check 1: Bound is strictly positive
    if bound_info['lower_bound'] > 0.001:
        print("✓ Branch bound is STRICTLY POSITIVE")
    else:
        print("✗ Branch bound is too close to zero")
    
    # Check 2: Bound is below empirical KL
    if bound_info['lower_bound'] <= kl_direct:
        print("✓ Branch bound ≤ empirical KL (valid lower bound)")
    else:
        print("✗ Branch bound > empirical KL (invalid!)")
    
    # Check 3: At least one branch has significant mass difference
    max_diff = max(abs(bound_info['branch_probs_p'][j] - bound_info['branch_probs_q'][j]) 
                   for j in range(3))
    if max_diff > 0.1:
        print(f"✓ Significant mode collapse detected (max |p(Aj) - q(Aj)| = {max_diff:.4f})")
    else:
        print(f"✗ No significant mode collapse (max diff = {max_diff:.4f})")
    
    print("\n" + "="*70)
    
    return target, model, bound_info, kl_direct


if __name__ == "__main__":
    target, model, bound_info, kl_direct = run_branch_collapse_experiment()