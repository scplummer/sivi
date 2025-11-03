"""
Experiment 7B: Multivariate BvM Coverage Transfer for Logistic Regression

Tests SIVI's ability to learn asymptotic posterior coverage in high dimensions.

PHASE 1: dim=5, n ∈ {50, 100, 200, 300}
PHASE 2: dim=20, n ∈ {200, 500, 1000} (if Phase 1 succeeds)

Key architecture principles (from working code):
- Depth SHRINKS as n grows (enforces Gaussian structure in BvM regime)
- Width stays ≥ dim (avoids rank collapse)
- σ_max ~ c/√n (critical guardrail against over-dispersion)
- Early stopping based on dispersion ratio convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Configuration
# ============================================================================

PHASE1_CONFIG = {
    'dim': 5,
    'sample_sizes': [50, 100, 200, 300],
    'n_replications': 50,
    'name': 'Phase1_dim5'
}

#======================================================================
#PHASE 1 SUMMARY
#======================================================================
#  n=  50: coverage=0.960, mean_err=17.15%, var_ratio=1.167
#  n= 100: coverage=0.960, mean_err=6.92%, var_ratio=1.072
#  n= 200: coverage=0.980, mean_err=3.59%, var_ratio=1.041
#  n= 300: coverage=0.960, mean_err=2.42%, var_ratio=1.027

PHASE2_CONFIG = {
    'dim': 20,
    'sample_sizes': [200, 500, 1000],
    'n_replications': 50,
    'name': 'Phase2_dim20'
}

#======================================================================
#PHASE 2 SUMMARY
#======================================================================
#  n= 200: coverage=0.980, mean_err=17.47%, var_ratio=1.144
#  n= 500: coverage=1.000, mean_err=6.30%, var_ratio=1.055
#  n=1000: coverage=0.920, mean_err=3.03%, var_ratio=1.026

# ============================================================================
# BvM-Aware Hyperparameters (from working code)
# ============================================================================

def get_bvm_hyperparams(n, dim):
    """
    SIMPLIFIED STRATEGY:
    1. Use StandardSIVI only for small n (< 100) where posterior may be non-Gaussian
    2. Switch to LinearGaussianSIVI early (n ≥ 100) for stability
    3. Keep σ_max loose and consistent - let the model learn naturally
    """
    
    # Architecture (only matters for StandardSIVI)
    if n < 100:
        hidden_dim = max(16, dim)  # Wider for flexibility
        depth = 2
    else:
        # Doesn't matter much since we'll use LinearGaussianSIVI
        hidden_dim = max(8, dim)
        depth = 1
    
    # Sigma bounds: SIMPLE and CONSISTENT
    if n >= 100:
        # Loose bound: ~2-3× true posterior std
        # True std ~ 1/√n, so set σ_max ~ 2/√n
        sigma_max = 2.0 / np.sqrt(n)
    else:
        sigma_max = 10.0  # Very loose for small n
    sigma_min = 0.005
    
    # Training: More steps for everyone
    if n < 100:
        steps = 800
    elif n < 200:
        steps = 1500
    else:
        steps = 2000
    
    K_train = min(256, n)  # Larger K for stability
    K_eval = max(1000, n)  # Even larger for evaluation
    lr = 1e-3
    
    # KEY: Use LinearGaussianSIVI early!
    use_linear_head = (n >= 100)  # Much earlier!
    
    return {
        'hidden_dim': hidden_dim,
        'depth': depth,
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
        'steps': steps,
        'K_train': K_train,
        'K_eval': K_eval,
        'lr': lr,
        'use_linear_head': use_linear_head
    }

# ============================================================================
# SIVI Models (from working code)
# ============================================================================

class StandardSIVI(nn.Module):
    """Standard SIVI with diagonal covariance and bounded sigma"""
    
    def __init__(self, latent_dim, param_dim, hidden_dim=64, depth=2,
                 sigma_min=0.01, sigma_max=5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Mean network
        layers_mu = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers_mu.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_mu.append(nn.Linear(hidden_dim, param_dim))
        self.mu_net = nn.Sequential(*layers_mu)
        
        # Log std network
        layers_sigma = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers_sigma.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_sigma.append(nn.Linear(hidden_dim, param_dim))
        self.log_sigma_net = nn.Sequential(*layers_sigma)
        
        # Base distribution: z ~ N(0, I)
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
    
    def forward(self, theta, K=128):
        """Compute log q(theta) using K importance samples"""
        batch_size = theta.shape[0]
        
        # Sample z ~ N(0, I)
        base_dist = torch.distributions.Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))  # [batch, K, latent_dim]
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        # Conditional parameters q(θ|z)
        mu = self.mu_net(z_flat)
        log_sigma = self.log_sigma_net(z_flat)
        sigma = torch.exp(log_sigma).clamp(min=self.sigma_min, max=self.sigma_max)
        
        mu = mu.reshape(batch_size, K, self.param_dim)
        sigma = sigma.reshape(batch_size, K, self.param_dim)
        
        # Expand theta: [batch, 1, param_dim]
        theta_expanded = theta.unsqueeze(1).expand(-1, K, -1)
        
        # Log p(θ|z) for each z
        log_prob_theta_given_z = -0.5 * torch.sum(
            ((theta_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        
        # Marginal: log q(θ) = log mean_z p(θ|z)
        log_q_theta = torch.logsumexp(log_prob_theta_given_z, dim=1) - np.log(K)
        
        return log_q_theta
    
    def sample(self, n_samples, K=1):
        """Sample from q(θ)"""
        base_dist = torch.distributions.Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((n_samples * K,))
        
        mu = self.mu_net(z)
        log_sigma = self.log_sigma_net(z)
        sigma = torch.exp(log_sigma).clamp(min=self.sigma_min, max=self.sigma_max)
        
        eps = torch.randn_like(mu)
        samples = mu + sigma * eps
        
        return samples
    
    def mean_and_cov(self, n_samples=10000):
        """Compute empirical mean and covariance"""
        with torch.no_grad():
            samples = self.sample(n_samples)
            mean = samples.mean(dim=0)
            cov = torch.cov(samples.T)
        return mean, cov


class LinearGaussianSIVI(nn.Module):
    """
    Affine SIVI: guarantees exact Gaussian marginal q(θ) = N(b, AA^T + Σ)
    Use for n ≥ 500 to eliminate mixture over-dispersion
    """
    
    def __init__(self, latent_dim, param_dim, sigma_min=0.01, sigma_max=5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Affine mean: μ(z) = Az + b
        self.A = nn.Parameter(torch.randn(param_dim, latent_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(param_dim))
        
        # Constant diagonal covariance
        init_sigma = (sigma_min + sigma_max) / 2
        self.log_diag = nn.Parameter(torch.log(torch.ones(param_dim) * init_sigma))
        
        # Base distribution
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
        
        self._update_marginal_params()
    
    def _update_marginal_params(self):
        """Update cached marginal parameters"""
        sigma = torch.exp(self.log_diag).clamp(min=self.sigma_min, max=self.sigma_max)
        self.marginal_cov = self.A @ self.A.T + torch.diag(sigma ** 2)
        self.marginal_mean = self.b
    
    def forward(self, theta, K=128):
        """Compute log q(theta) using exact Gaussian"""
        self._update_marginal_params()
        
        diff = theta - self.marginal_mean
        
        try:
            prec_diff = torch.linalg.solve(self.marginal_cov, diff.T).T
            log_prob = -0.5 * torch.sum(diff * prec_diff, dim=-1)
            log_prob -= 0.5 * torch.logdet(self.marginal_cov)
            log_prob -= self.param_dim * 0.5 * np.log(2 * np.pi)
        except:
            # Ridge for stability
            cov_ridge = self.marginal_cov + 1e-6 * torch.eye(self.param_dim).to(theta.device)
            prec_diff = torch.linalg.solve(cov_ridge, diff.T).T
            log_prob = -0.5 * torch.sum(diff * prec_diff, dim=-1)
            log_prob -= 0.5 * torch.logdet(cov_ridge)
            log_prob -= self.param_dim * 0.5 * np.log(2 * np.pi)
        
        return log_prob
    
    def sample(self, n_samples, K=1):
        """Sample from N(b, AA^T + Σ)"""
        self._update_marginal_params()
        
        try:
            L = torch.linalg.cholesky(self.marginal_cov)
        except:
            cov_ridge = self.marginal_cov + 1e-6 * torch.eye(self.param_dim).to(self.marginal_cov.device)
            L = torch.linalg.cholesky(cov_ridge)
        
        eps = torch.randn(n_samples, self.param_dim).to(L.device)
        samples = self.marginal_mean + (L @ eps.T).T
        
        return samples
    
    def mean_and_cov(self, n_samples=10000):
        """Return exact mean and covariance"""
        self._update_marginal_params()
        return self.marginal_mean.detach(), self.marginal_cov.detach()


# ============================================================================
# Logistic Regression Model
# ============================================================================

class LogisticRegression:
    """
    Y_i ~ Bernoulli(σ(x_i^T θ)), θ ~ N(0, τ² I)
    
    BvM: Posterior → N(θ_MLE, [X^T W X + I/τ²]^(-1))
    where W = diag(p_i(1-p_i))
    """
    
    def __init__(self, dim, prior_tau=5.0):
        self.dim = dim
        self.prior_tau = prior_tau
    
    def sample_data(self, n, theta_true):
        """Generate n observations with random design matrix"""
        X = torch.randn(n, self.dim)
        logits = X @ theta_true
        probs = torch.sigmoid(logits)
        y = torch.bernoulli(probs).long()
        return X, y
    
    def log_posterior(self, theta, X, y):
        """Log p(θ|X,y) ∝ log p(y|X,θ) + log p(θ)"""
        # Log likelihood
        logits = X @ theta
        log_lik = torch.sum(y * logits - torch.log(1 + torch.exp(logits)))
        
        # Log prior
        log_prior = -0.5 * torch.sum(theta ** 2) / (self.prior_tau ** 2)
        log_prior -= self.dim * 0.5 * np.log(2 * np.pi * self.prior_tau ** 2)
        
        return log_lik + log_prior
    
    def laplace_approximation(self, X, y, ridge=1e-6):
        """
        Compute Laplace approximation: MLE and Fisher information
        Returns: θ_MLE, Σ_Laplace
        """
        theta = torch.zeros(self.dim)
        
        # Newton's method for MLE
        for _ in range(50):
            logits = X @ theta
            probs = torch.sigmoid(logits)
            
            # Gradient
            grad = X.T @ (y - probs) - theta / (self.prior_tau ** 2)
            
            # Hessian (negative definite)
            W = probs * (1 - probs)
            H = -X.T @ (W.unsqueeze(1) * X) - torch.eye(self.dim) / (self.prior_tau ** 2)
            H_ridge = H - ridge * torch.eye(self.dim)
            
            # Newton step
            try:
                delta = torch.linalg.solve(H_ridge, grad)
                theta = theta - delta
                if torch.norm(delta) < 1e-6:
                    break
            except:
                break
        
        # Fisher information and covariance
        logits = X @ theta
        probs = torch.sigmoid(logits)
        W = probs * (1 - probs)
        Fisher = X.T @ (W.unsqueeze(1) * X) + torch.eye(self.dim) / (self.prior_tau ** 2)
        Fisher_ridge = Fisher + ridge * torch.eye(self.dim)
        
        Sigma_Laplace = torch.linalg.inv(Fisher_ridge)
        
        return theta, Sigma_Laplace


# ============================================================================
# Training (from working code)
# ============================================================================

def train_sivi(model, X, y, logistic_model, hypers, verbose=True):
    """Train SIVI with SMART INITIALIZATION near Laplace approximation"""
    
    # Get Laplace approximation
    theta_laplace, Sigma_laplace = logistic_model.laplace_approximation(X, y)
    sigma_laplace = torch.sqrt(torch.diag(Sigma_laplace))
    
    # SMART INITIALIZATION (NEW!)
    if isinstance(model, LinearGaussianSIVI):
        # Initialize b ≈ θ_MLE, log_diag ≈ log(σ_Laplace)
        with torch.no_grad():
            model.b.copy_(theta_laplace)
            model.log_diag.copy_(torch.log(sigma_laplace))
            # A stays random (captures covariance structure)
    else:
        # For StandardSIVI, initialize final layers to output Laplace params
        with torch.no_grad():
            # Mean network: output θ_MLE on average
            if hasattr(model.mu_net, '0'):  # Check if Sequential
                model.mu_net[-1].bias.copy_(theta_laplace)
                model.mu_net[-1].weight.normal_(0, 0.01)
            
            # Sigma network: output σ_Laplace on average  
            if hasattr(model.log_sigma_net, '0'):
                model.log_sigma_net[-1].bias.copy_(torch.log(sigma_laplace))
                model.log_sigma_net[-1].weight.normal_(0, 0.01)
    
    optimizer = optim.Adam(model.parameters(), lr=hypers['lr'], weight_decay=1e-4)
    
    # Longer warmup for large n
    warmup_steps = min(200, hypers['steps'] // 5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hypers['steps'] - warmup_steps, eta_min=1e-5
    )
    
    X_dev = X.to(device)
    y_dev = y.to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    pbar = tqdm(range(hypers['steps']), desc='Training', leave=False, disable=not verbose)
    
    for step in pbar:
        # Learning rate warmup
        if step < warmup_steps:
            lr = hypers['lr'] * (step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Sample from q
        theta_samples = model.sample(hypers['K_train']).to(device)
        
        # Compute log q(θ)
        log_q = model(theta_samples, K=hypers['K_train'])
        
        # Compute log p(θ|X,y)
        log_post = torch.stack([
            logistic_model.log_posterior(theta_samples[i], X_dev, y_dev)
            for i in range(len(theta_samples))
        ])
        
        # Forward KL
        loss = (log_q - log_post).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step >= warmup_steps:
            scheduler.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Early stopping: STRICTER criteria
        if step % 50 == 0 and step > 300:  # Wait longer before early stop
            with torch.no_grad():
                sivi_mean, sivi_cov = model.mean_and_cov(n_samples=2000)  # More samples
                
                recent_loss = np.mean(losses[-20:])  # Longer window
                loss_improved = (best_loss - recent_loss) / abs(best_loss) > 0.0005  # Stricter
                
                # Stricter dispersion check
                disp_ratio = torch.trace(sivi_cov @ torch.linalg.inv(Sigma_laplace)) / logistic_model.dim
                dispersion_ok = 0.90 <= disp_ratio.item() <= 1.12  # TIGHTER from [0.90, 1.10]
                
                # Also check mean accuracy
                mean_error = torch.norm(sivi_mean - theta_laplace) / torch.norm(theta_laplace)
                mean_ok = mean_error.item() < 0.05  # Within 5%
                
                if not loss_improved and dispersion_ok and mean_ok:
                    patience_counter += 1
                    if patience_counter >= 3:  # More patience
                        if verbose:
                            print(f"  Early stop at step {step}")
                        break
                else:
                    patience_counter = 0
                    best_loss = min(best_loss, recent_loss)
    
    pbar.close()
    return model

# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def compute_coverage_and_diagnostics(sivi, theta_true, theta_laplace, Sigma_laplace, 
                                     dim, K_eval, alpha=0.05):
    """
    Compute coverage and approximation quality metrics
    
    Returns:
        inside: bool, whether θ_true is in (1-α) credible ellipsoid
        diagnostics: dict with mean_error, var_ratio, etc.
    """
    # Get SIVI approximation
    sivi_mean, sivi_cov = sivi.mean_and_cov(n_samples=K_eval)
    sivi_mean = sivi_mean.cpu()
    sivi_cov = sivi_cov.cpu()
    
    # Coverage: Mahalanobis distance test
    delta = theta_true - sivi_mean
    try:
        dist_sq = delta @ torch.linalg.solve(sivi_cov, delta)
    except:
        sivi_cov_ridge = sivi_cov + 1e-6 * torch.eye(dim)
        dist_sq = delta @ torch.linalg.solve(sivi_cov_ridge, delta)
    
    threshold = chi2.ppf(1 - alpha, df=dim)
    inside = (dist_sq <= threshold).item()
    
    # Diagnostics
    mean_error = torch.norm(sivi_mean - theta_laplace).item()
    rel_mean_error = mean_error / torch.norm(theta_laplace).item() if torch.norm(theta_laplace) > 1e-6 else mean_error
    
    var_ratio = (torch.diag(sivi_cov) / torch.diag(Sigma_laplace)).mean().item()
    cov_error = torch.norm(sivi_cov - Sigma_laplace, p='fro').item()
    
    # Rank check (for StandardSIVI only)
    if hasattr(sivi, 'mu_net'):
        z_samples = torch.randn(100, sivi.latent_dim)
        mu_samples = sivi.mu_net(z_samples)
        
        z_centered = z_samples - z_samples.mean(dim=0)
        mu_centered = mu_samples - mu_samples.mean(dim=0)
        J_approx = (mu_centered.T @ z_centered) / len(z_samples)
        
        try:
            singular_values = torch.linalg.svdvals(J_approx)
            min_singular_value = singular_values.min().item()
        except:
            min_singular_value = 0.0
    else:
        min_singular_value = float('nan')
    
    diagnostics = {
        'mean_error': rel_mean_error,
        'var_ratio': var_ratio,
        'cov_error': cov_error,
        'min_singular_value': min_singular_value,
        'dist_sq': dist_sq.item(),
        'threshold': threshold
    }
    
    return inside, diagnostics


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(config):
    """
    Run BvM coverage experiment for given configuration
    
    Args:
        config: dict with 'dim', 'sample_sizes', 'n_replications', 'name'
    """
    print(f"\n{'='*70}")
    print(f"{config['name']}: dim={config['dim']}")
    print(f"Sample sizes: {config['sample_sizes']}")
    print(f"{'='*70}")
    
    dim = config['dim']
    sample_sizes = config['sample_sizes']
    R = config['n_replications']
    alpha = 0.05
    
    # True parameter
    theta_true = torch.randn(dim) * 0.5
    prior_tau = 5.0
    
    logistic_model = LogisticRegression(dim=dim, prior_tau=prior_tau)
    
    results = []
    
    for n in sample_sizes:
        print(f"\n{'-'*70}")
        print(f"n = {n} (n/dim = {n/dim:.1f})")
        print(f"{'-'*70}")
        
        # Get hyperparameters
        hypers = get_bvm_hyperparams(n, dim)
        
        print(f"  Architecture: hidden={hypers['hidden_dim']}, depth={hypers['depth']}")
        print(f"  σ bounds: [{hypers['sigma_min']:.4f}, {hypers['sigma_max']:.4f}]")
        print(f"  Training: {hypers['steps']} steps, K_train={hypers['K_train']}")
        print(f"  Linear head: {hypers['use_linear_head']}")
        
        covers = []
        diagnostics_list = []
        
        for r in tqdm(range(R), desc=f'Replications'):
            # Generate data
            X, y = logistic_model.sample_data(n, theta_true)
            
            # Laplace approximation (ground truth)
            theta_laplace, Sigma_laplace = logistic_model.laplace_approximation(X, y)
            
            # Create SIVI model
            if hypers['use_linear_head']:
                sivi = LinearGaussianSIVI(
                    latent_dim=dim,
                    param_dim=dim,
                    sigma_min=hypers['sigma_min'],
                    sigma_max=hypers['sigma_max']
                ).to(device)
            else:
                sivi = StandardSIVI(
                    latent_dim=dim,
                    param_dim=dim,
                    hidden_dim=hypers['hidden_dim'],
                    depth=hypers['depth'],
                    sigma_min=hypers['sigma_min'],
                    sigma_max=hypers['sigma_max']
                ).to(device)
                
                # Xavier init
                for m in sivi.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight, gain=0.5)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            # Train
            sivi = train_sivi(sivi, X, y, logistic_model, hypers, verbose=(r == 0))
            sivi.eval()
            
            # Evaluate
            inside, diag = compute_coverage_and_diagnostics(
                sivi, theta_true, theta_laplace, Sigma_laplace,
                dim, hypers['K_eval'], alpha
            )
            
            covers.append(inside)
            diagnostics_list.append(diag)
            
            if r == 0:
                print(f"\n  Sample diagnostics (rep 0):")
                print(f"    Mean error: {diag['mean_error']:.2%}")
                print(f"    Var ratio: {diag['var_ratio']:.3f}")
                if not np.isnan(diag['min_singular_value']):
                    print(f"    Min singular value: {diag['min_singular_value']:.4f}")
        
        # Aggregate
        cov_rate = np.mean(covers)
        cov_sd = np.sqrt(cov_rate * (1 - cov_rate) / R)
        
        mean_errors = [d['mean_error'] for d in diagnostics_list]
        var_ratios = [d['var_ratio'] for d in diagnostics_list]
        
        print(f"\n  Aggregate (R={R}):")
        print(f"    Coverage: {cov_rate:.3f} ± {cov_sd:.3f} (nominal: {1-alpha:.3f})")
        print(f"    Mean error: {np.mean(mean_errors):.2%} ± {np.std(mean_errors):.2%}")
        print(f"    Var ratio: {np.mean(var_ratios):.3f} ± {np.std(var_ratios):.3f}")
        
        results.append({
            'n': n,
            'coverage': cov_rate,
            'coverage_sd': cov_sd,
            'mean_error': np.mean(mean_errors),
            'mean_error_sd': np.std(mean_errors),
            'var_ratio': np.mean(var_ratios),
            'var_ratio_sd': np.std(var_ratios)
        })
    
    # Plot
    plot_results(results, config, alpha)
    
    return results


def plot_results(results, config, alpha):
    """Create diagnostic plots"""
    N_list = [r['n'] for r in results]
    coverages = [r['coverage'] for r in results]
    coverage_sds = [r['coverage_sd'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    var_ratios = [r['var_ratio'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Coverage
    ax = axes[0]
    ax.errorbar(N_list, coverages, yerr=coverage_sds, fmt='o-', capsize=5,
                linewidth=2, markersize=8, label='SIVI Coverage', color='purple')
    ax.axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, 
               label=f'Nominal {1-alpha:.0%}')
    R = config['n_replications']
    ax.fill_between(N_list,
                     [1-alpha - 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     [1-alpha + 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     alpha=0.2, color='red', label='95% Binomial CI')
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title(f'{config["name"]}: Credible Coverage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.80, 1.0])
    
    # Mean error
    ax = axes[1]
    ax.plot(N_list, mean_errors, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Relative Mean Error', fontsize=12)
    ax.set_title('Mean Approximation Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Variance ratio
    ax = axes[2]
    ax.plot(N_list, var_ratios, 'o-', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(N_list, [0.95]*len(N_list), [1.05]*len(N_list), alpha=0.2, color='gray')
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Variance Ratio (SIVI/Laplace)', fontsize=12)
    ax.set_title('Variance Approximation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.20])
    
    plt.tight_layout()
    
    filename = f'{config["name"]}_coverage.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n  Plot saved: {filename}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTIVARIATE BvM COVERAGE TRANSFER")
    print("="*70)
    
    # Phase 1: dim=5
    print("\n" + "="*70)
    print("PHASE 1: dim=5 Logistic Regression")
    print("="*70)
    
    results_phase1 = run_experiment(PHASE1_CONFIG)
    
    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*70}")
    for r in results_phase1:
        print(f"  n={r['n']:4d}: coverage={r['coverage']:.3f}, "
              f"mean_err={r['mean_error']:.2%}, var_ratio={r['var_ratio']:.3f}")
    
    # Check if Phase 1 succeeded
    final_n = results_phase1[-1]
    success = (final_n['coverage'] > 0.90 and 
               final_n['mean_error'] < 0.20 and
               0.85 < final_n['var_ratio'] < 1.15)
    
    if success:
        print("\n✓ Phase 1 SUCCEEDED! Proceeding to Phase 2...")
        
        # Phase 2: dim=20
        print("\n" + "="*70)
        print("PHASE 2: dim=20 Logistic Regression")
        print("="*70)
        
        results_phase2 = run_experiment(PHASE2_CONFIG)
        
        print(f"\n{'='*70}")
        print("PHASE 2 SUMMARY")
        print(f"{'='*70}")
        for r in results_phase2:
            print(f"  n={r['n']:4d}: coverage={r['coverage']:.3f}, "
                  f"mean_err={r['mean_error']:.2%}, var_ratio={r['var_ratio']:.3f}")
    else:
        print("\n✗ Phase 1 needs tuning before Phase 2")
        print("Consider:")
        print("  - Longer training (increase steps)")
        print("  - Adjust σ_max scaling constant")
        print("  - Different learning rate schedule")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)