"""
Experiment 7: Coverage Transfer / Bernstein-von Mises Sanity Check

Goal: Verify that credible sets from SIVI achieve near-nominal frequentist coverage
in regular models as n increases, matching the Bernstein-von Mises theorem (Gaussian posterior).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal, Bernoulli
from scipy.stats import chi2, norm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# SIVI Model for Posterior Approximation
# ============================================================================

class PosteriorSIVI(nn.Module):
    """SIVI for approximating posterior distributions"""
    def __init__(self, latent_dim=2, param_dim=2, hidden_dim=64, depth=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        
        # MLP for mean
        layers_mu = []
        layers_mu.append(nn.Linear(latent_dim, hidden_dim))
        layers_mu.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers_mu.append(nn.Linear(hidden_dim, hidden_dim))
            layers_mu.append(nn.ReLU())
        
        layers_mu.append(nn.Linear(hidden_dim, param_dim))
        self.mu_net = nn.Sequential(*layers_mu)
        
        # MLP for log std (tail-safe: bounded sigma)
        layers_sigma = []
        layers_sigma.append(nn.Linear(latent_dim, hidden_dim))
        layers_sigma.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers_sigma.append(nn.Linear(hidden_dim, hidden_dim))
            layers_sigma.append(nn.ReLU())
        
        layers_sigma.append(nn.Linear(hidden_dim, param_dim))
        self.log_sigma_net = nn.Sequential(*layers_sigma)
        
        # Base distribution
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
    
    def forward(self, theta, K=128):
        """Compute log q(theta) using K samples"""
        batch_size = theta.shape[0]
        
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        mu = self.mu_net(z_flat)
        log_sigma = self.log_sigma_net(z_flat)
        # Bounded sigma for tail safety
        sigma = torch.exp(log_sigma).clamp(min=0.01, max=5.0)
        
        mu = mu.reshape(batch_size, K, self.param_dim)
        sigma = sigma.reshape(batch_size, K, self.param_dim)
        
        theta_expanded = theta.unsqueeze(1).expand(-1, K, -1)
        
        log_prob_theta_z = -0.5 * torch.sum(
            ((theta_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        
        log_q_theta = torch.logsumexp(log_prob_theta_z, dim=1) - np.log(K)
        
        return log_q_theta
    
    def sample(self, n_samples, K=1):
        """Sample from the SIVI distribution"""
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((n_samples * K,))
        
        mu = self.mu_net(z)
        log_sigma = self.log_sigma_net(z)
        sigma = torch.exp(log_sigma).clamp(min=0.01, max=5.0)
        
        eps = torch.randn_like(mu)
        samples = mu + sigma * eps
        
        return samples
    
    def mean_and_cov(self, n_samples=10000):
        """Compute mean and covariance from samples"""
        with torch.no_grad():
            samples = self.sample(n_samples)
            mean = samples.mean(dim=0)
            cov = torch.cov(samples.T)
        return mean, cov


# ============================================================================
# Model 1: Gaussian Mean Estimation
# ============================================================================

class GaussianMeanModel:
    """
    Y_i ~ N(theta, I), theta ~ N(0, tau^2 I)
    
    Conjugate: posterior is N(theta_post, Sigma_post) where
    Sigma_post = (I/sigma^2 + n*I)^{-1} = (n + 1)^{-1} I  (with tau=1, sigma=1)
    theta_post = Sigma_post @ (0 + n * ybar) = n/(n+1) * ybar
    """
    def __init__(self, dim=2, prior_tau=1.0, likelihood_sigma=1.0):
        self.dim = dim
        self.prior_tau = prior_tau
        self.likelihood_sigma = likelihood_sigma
    
    def sample_data(self, n, theta_true):
        """Generate n observations from N(theta_true, I)"""
        return theta_true + torch.randn(n, self.dim)
    
    def conjugate_posterior(self, y):
        """Exact posterior parameters (conjugate)"""
        n = len(y)
        ybar = y.mean(dim=0)
        
        # Posterior precision: prior_precision + n * likelihood_precision
        prior_prec = 1.0 / (self.prior_tau ** 2)
        lik_prec = 1.0 / (self.likelihood_sigma ** 2)
        post_prec = prior_prec + n * lik_prec
        
        # Posterior covariance
        post_cov = torch.eye(self.dim) / post_prec
        
        # Posterior mean
        post_mean = (n * lik_prec / post_prec) * ybar
        
        return post_mean, post_cov
    
    def log_posterior(self, theta, y):
        """
        log p(theta | y) ∝ log p(y | theta) + log p(theta)
        """
        n = len(y)
        
        # Log likelihood: sum_i log N(y_i | theta, sigma^2 I)
        diff = y - theta.unsqueeze(0)
        log_lik = -0.5 * torch.sum(diff ** 2) / (self.likelihood_sigma ** 2)
        log_lik -= n * self.dim * 0.5 * np.log(2 * np.pi * self.likelihood_sigma ** 2)
        
        # Log prior: log N(theta | 0, tau^2 I)
        log_prior = -0.5 * torch.sum(theta ** 2) / (self.prior_tau ** 2)
        log_prior -= self.dim * 0.5 * np.log(2 * np.pi * self.prior_tau ** 2)
        
        return log_lik + log_prior


# ============================================================================
# Model 2: Logistic Regression
# ============================================================================

class LogisticRegressionModel:
    """
    Y_i ~ Bern(sigma(x_i^T theta)), theta ~ N(0, tau^2 I)
    """
    def __init__(self, dim=5, prior_tau=5.0):
        self.dim = dim
        self.prior_tau = prior_tau
    
    def sample_data(self, n, theta_true):
        """Generate n observations from Bernoulli(logistic(X @ theta))"""
        # Random design matrix
        X = torch.randn(n, self.dim)
        
        # Logistic probabilities
        logits = X @ theta_true
        probs = torch.sigmoid(logits)
        
        # Sample binary outcomes
        y = torch.bernoulli(probs).long()
        
        return X, y
    
    def log_posterior(self, theta, X, y):
        """Log posterior: log p(theta | X, y) ∝ log p(y | X, theta) + log p(theta)"""
        # Log likelihood
        logits = X @ theta
        log_lik = torch.sum(y * logits - torch.log(1 + torch.exp(logits)))
        
        # Log prior
        log_prior = -0.5 * torch.sum(theta ** 2) / (self.prior_tau ** 2)
        log_prior -= self.dim * 0.5 * np.log(2 * np.pi * self.prior_tau ** 2)
        
        return log_lik + log_prior
    
    def mle_and_fisher(self, X, y, ridge=1e-6):
        """
        Compute MLE and observed Fisher information
        
        For logistic: I_obs = X^T W X where W = diag(p_i(1-p_i))
        """
        # Newton's method for MLE
        theta = torch.zeros(self.dim)
        
        for _ in range(50):
            logits = X @ theta
            probs = torch.sigmoid(logits)
            
            # Gradient
            grad = X.T @ (y - probs) - theta / (self.prior_tau ** 2)
            
            # Hessian (with ridge for stability)
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
        
        # Fisher information (negative Hessian at MLE)
        logits = X @ theta
        probs = torch.sigmoid(logits)
        W = probs * (1 - probs)
        Fisher = X.T @ (W.unsqueeze(1) * X) + torch.eye(self.dim) / (self.prior_tau ** 2)
        Fisher = Fisher + ridge * torch.eye(self.dim)  # Ridge for stability
        
        return theta, Fisher


# ============================================================================
# SIVI Posterior Fitting
# ============================================================================

def fit_sivi_posterior_gaussian(y, prior_tau=1.0, K=128, steps=3000, lr=1e-3):
    """Fit SIVI to Gaussian mean posterior using forward KL"""
    dim = y.shape[1]
    model_gauss = GaussianMeanModel(dim=dim, prior_tau=prior_tau)
    
    sivi = PosteriorSIVI(latent_dim=dim, param_dim=dim, hidden_dim=64, depth=2).to(device)
    
    # Initialize
    for m in sivi.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.3)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    optimizer = optim.Adam(sivi.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    
    y_dev = y.to(device)
    
    pbar = tqdm(range(steps), desc='Training SIVI', leave=False)
    for step in pbar:
        # Sample from q (current SIVI approximation)
        theta = sivi.sample(256).to(device)
        
        # Compute log q(theta)
        log_q = sivi(theta, K=K)
        
        # Compute log posterior p(theta|y)
        log_post = torch.stack([
            model_gauss.log_posterior(theta[i], y_dev) for i in range(len(theta))
        ])
        
        # Forward KL: E_q[log q(theta) - log p(theta|y)]
        loss = (log_q - log_post).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sivi.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    pbar.close()
    return sivi


def fit_sivi_posterior_logistic(X, y, prior_tau=5.0, K=128, steps=3000, lr=1e-3):
    """Fit SIVI to logistic regression posterior using forward KL"""
    dim = X.shape[1]
    model_logistic = LogisticRegressionModel(dim=dim, prior_tau=prior_tau)
    
    sivi = PosteriorSIVI(latent_dim=dim, param_dim=dim, hidden_dim=64, depth=2).to(device)
    
    # Initialize
    for m in sivi.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.3)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    optimizer = optim.Adam(sivi.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    
    X_dev = X.to(device)
    y_dev = y.to(device)
    
    pbar = tqdm(range(steps), desc='Training SIVI', leave=False)
    for step in pbar:
        # Sample from q (current SIVI approximation)
        theta = sivi.sample(256).to(device)
        
        # Compute log q(theta)
        log_q = sivi(theta, K=K)
        
        # Compute log posterior p(theta|X,y)
        log_post = torch.stack([
            model_logistic.log_posterior(theta[i], X_dev, y_dev) for i in range(len(theta))
        ])
        
        # Forward KL: E_q[log q(theta) - log p(theta|X,y)]
        loss = (log_q - log_post).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sivi.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    pbar.close()
    return sivi


# ============================================================================
# Credible Coverage
# ============================================================================

@torch.no_grad()
def credible_coverage(sivi, theta_true, alpha=0.05, M=5000):
    """
    Compute whether theta_true falls in the (1-alpha) credible ellipsoid
    
    Ellipsoid: {theta : (theta - m)^T Sigma^{-1} (theta - m) <= chi2_{d, 1-alpha}}
    """
    # Get mean and covariance
    m, S = sivi.mean_and_cov(n_samples=M)
    m = m.cpu()
    S = S.cpu()
    
    # Mahalanobis distance
    delta = theta_true - m
    
    try:
        dist_sq = delta @ torch.linalg.solve(S, delta)
    except:
        # If singular, add small ridge
        S_ridge = S + 1e-6 * torch.eye(len(S))
        dist_sq = delta @ torch.linalg.solve(S_ridge, delta)
    
    # Chi-squared threshold
    d = len(theta_true)
    threshold = chi2.ppf(1 - alpha, df=d)
    
    inside = (dist_sq <= threshold).item()
    
    return inside, dist_sq.item(), threshold


# ============================================================================
# BvM QQ Plot
# ============================================================================

@torch.no_grad()
def bvm_qq_test(sivi, theta_hat, Fisher, n, M=5000):
    """
    Test whether standardized SIVI samples follow N(0, I)
    
    Z = sqrt(n) * I^{1/2} * (theta - theta_hat)
    
    Returns QQ plot data
    """
    # Sample from SIVI
    theta_samples = sivi.sample(M).cpu()
    
    # Standardize
    L = torch.linalg.cholesky(Fisher.cpu())  # I^{1/2}
    Z = np.sqrt(n) * (theta_samples - theta_hat.cpu()) @ L.T
    
    # For multivariate, look at marginals
    d = Z.shape[1]
    
    qq_data = []
    for j in range(d):
        z_j = Z[:, j].detach().numpy()
        z_j_sorted = np.sort(z_j)
        
        # Theoretical quantiles
        quantiles = (np.arange(1, M+1) - 0.5) / M
        theoretical = norm.ppf(quantiles)
        
        qq_data.append((theoretical, z_j_sorted))
    
    return qq_data


def compute_w2_to_standard_normal(Z):
    """Compute average Wasserstein-2 distance to N(0,1) across dimensions"""
    d = Z.shape[1]
    w2_dists = []
    
    for j in range(d):
        z_j = Z[:, j].detach().cpu().numpy()
        # Sample from N(0,1)
        normal_samples = np.random.randn(len(z_j))
        
        # Sort both
        z_j_sorted = np.sort(z_j)
        normal_sorted = np.sort(normal_samples)
        
        # W2 distance (L2 distance between quantile functions)
        w2 = np.sqrt(np.mean((z_j_sorted - normal_sorted) ** 2))
        w2_dists.append(w2)
    
    return np.mean(w2_dists)


# ============================================================================
# Main Experiments
# ============================================================================

def run_gaussian_mean_experiment():
    """
    Experiment: Gaussian mean estimation with increasing n
    Check coverage and BvM convergence
    """
    print("="*70)
    print("GAUSSIAN MEAN MODEL: Coverage & BvM Test")
    print("="*70)
    
    dim = 2
    theta_true = torch.tensor([1.0, -0.5])
    prior_tau = 1.0
    
    N_list = [50, 100, 200, 500, 1000]
    R = 10  # Replications
    alpha = 0.05
    
    coverage_rates = []
    coverage_sds = []
    w2_distances = []
    
    for n in N_list:
        print(f"\n--- n = {n} ---")
        
        covers = []
        w2_list = []
        
        for r in tqdm(range(R), desc=f'Replications (n={n})'):
            # Generate data
            model = GaussianMeanModel(dim=dim, prior_tau=prior_tau)
            y = model.sample_data(n, theta_true)
            
            # Fit SIVI
            sivi = fit_sivi_posterior_gaussian(y, prior_tau=prior_tau, K=128, steps=5000, lr=1e-3)
            sivi.eval()
            
            # Check coverage
            inside, _, _ = credible_coverage(sivi, theta_true, alpha=alpha, M=5000)
            covers.append(inside)
            
            # BvM test (for large n)
            if n >= 200:
                # Conjugate posterior mean as MLE proxy
                post_mean, post_cov = model.conjugate_posterior(y)
                Fisher = torch.linalg.inv(post_cov) * n  # Scale by n
                
                qq_data = bvm_qq_test(sivi, post_mean, Fisher, n, M=5000)
                
                # Compute W2 for first dimension
                theta_samples = sivi.sample(5000).cpu()
                L = torch.linalg.cholesky(Fisher)
                Z = np.sqrt(n) * (theta_samples - post_mean) @ L.T
                w2 = compute_w2_to_standard_normal(Z)
                w2_list.append(w2)
        
        # Statistics
        cov_rate = np.mean(covers)
        cov_sd = np.sqrt(cov_rate * (1 - cov_rate) / R)
        coverage_rates.append(cov_rate)
        coverage_sds.append(cov_sd)
        
        if w2_list:
            w2_distances.append(np.mean(w2_list))
        else:
            w2_distances.append(np.nan)
        
        print(f"  Coverage: {cov_rate:.3f} ± {cov_sd:.3f} (nominal: {1-alpha:.3f})")
        if w2_list:
            print(f"  W2 to N(0,1): {np.mean(w2_list):.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Coverage plot
    ax1.errorbar(N_list, coverage_rates, yerr=coverage_sds, fmt='o-', capsize=5,
                linewidth=2, markersize=8, label='SIVI Coverage')
    ax1.axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, label=f'Nominal {1-alpha:.0%}')
    ax1.fill_between(N_list, 
                     [1-alpha - 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     [1-alpha + 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     alpha=0.2, color='red', label='95% Binomial CI')
    ax1.set_xlabel('Sample Size n', fontsize=12)
    ax1.set_ylabel('Coverage Rate', fontsize=12)
    ax1.set_title('Credible Coverage vs Sample Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.85, 1.0])
    
    # W2 plot
    valid_idx = ~np.isnan(w2_distances)
    if np.any(valid_idx):
        ax2.plot(np.array(N_list)[valid_idx], np.array(w2_distances)[valid_idx], 
                'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Sample Size n', fontsize=12)
        ax2.set_ylabel('W₂ Distance to N(0,1)', fontsize=12)
        ax2.set_title('BvM Convergence (standardized draws)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gaussian_mean_coverage.png', dpi=300)
    plt.show()
    
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA - Gaussian Mean")
    print("="*70)
    
    # Check convergence to nominal
    if len(coverage_rates) >= 3:
        final_cov = coverage_rates[-1]
        if abs(final_cov - (1-alpha)) < 0.03:
            print(f"✓ Coverage at n={N_list[-1]} within 3% of nominal")
        else:
            print(f"✗ Coverage deviation: {abs(final_cov - (1-alpha)):.3f}")
    
    # Check W2 decreasing
    if len([w for w in w2_distances if not np.isnan(w)]) >= 2:
        w2_valid = [w for w in w2_distances if not np.isnan(w)]
        if w2_valid[-1] < w2_valid[0]:
            print(f"✓ W₂ distance decreased from {w2_valid[0]:.4f} to {w2_valid[-1]:.4f}")
        else:
            print(f"✗ W₂ distance not decreasing")
    
    return N_list, coverage_rates, coverage_sds, w2_distances


def run_logistic_experiment():
    """
    Experiment: Logistic regression with increasing n
    Check coverage improvement
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION MODEL: Coverage Test")
    print("="*70)
    
    dim = 5
    theta_true = torch.randn(dim) * 0.5  # Moderate coefficients
    prior_tau = 5.0
    
    N_list = [100, 200, 500, 1000]
    R = 5  # Fewer replications (slower)
    alpha = 0.05
    
    coverage_rates = []
    coverage_sds = []
    
    for n in N_list:
        print(f"\n--- n = {n} ---")
        
        covers = []
        
        for r in tqdm(range(R), desc=f'Replications (n={n})'):
            # Generate data
            model = LogisticRegressionModel(dim=dim, prior_tau=prior_tau)
            X, y = model.sample_data(n, theta_true)
            
            # Fit SIVI
            sivi = fit_sivi_posterior_logistic(X, y, prior_tau=prior_tau, 
                                               K=128, steps=5000, lr=1e-3)
            sivi.eval()
            
            # Check coverage
            inside, _, _ = credible_coverage(sivi, theta_true, alpha=alpha, M=3000)
            covers.append(inside)
        
        # Statistics
        cov_rate = np.mean(covers)
        cov_sd = np.sqrt(cov_rate * (1 - cov_rate) / R)
        coverage_rates.append(cov_rate)
        coverage_sds.append(cov_sd)
        
        print(f"  Coverage: {cov_rate:.3f} ± {cov_sd:.3f} (nominal: {1-alpha:.3f})")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(N_list, coverage_rates, yerr=coverage_sds, fmt='s-', capsize=5,
                linewidth=2, markersize=8, label='SIVI Coverage', color='purple')
    plt.axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, label=f'Nominal {1-alpha:.0%}')
    plt.fill_between(N_list, 
                     [1-alpha - 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     [1-alpha + 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     alpha=0.2, color='red', label='95% Binomial CI')
    plt.xlabel('Sample Size n', fontsize=12)
    plt.ylabel('Coverage Rate', fontsize=12)
    plt.title('Logistic Regression: Credible Coverage vs Sample Size', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.7, 1.0])
    plt.tight_layout()
    plt.savefig('logistic_coverage.png', dpi=300)
    plt.show()
    
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA - Logistic Regression")
    print("="*70)
    
    # Check improvement with n
    if coverage_rates[-1] > coverage_rates[0]:
        print(f"✓ Coverage improved from {coverage_rates[0]:.3f} to {coverage_rates[-1]:.3f}")
    else:
        print(f"✗ Coverage did not improve")
    
    # Check reasonable coverage at large n
    if coverage_rates[-1] > 0.85:
        print(f"✓ Coverage at n={N_list[-1]} is reasonable (> 85%)")
    else:
        print(f"✗ Coverage at large n is too low")
    
    return N_list, coverage_rates, coverage_sds


# ============================================================================
# Main runner
# ============================================================================

if __name__ == "__main__":
    # Run Gaussian mean experiment
    print("Starting Gaussian Mean Experiment...")
    gauss_results = run_gaussian_mean_experiment()
    
    # Run logistic experiment
    print("\n\nStarting Logistic Regression Experiment...")
    logistic_results = run_logistic_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT 7 COMPLETE")
    print("="*70)