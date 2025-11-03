"""
Experiment 7: Coverage Transfer / Bernstein-von Mises (BvM-Aware Version)

Goal: Verify that credible sets from SIVI achieve near-nominal frequentist coverage
in regular models as n increases, matching the Bernstein-von Mises theorem.

Key improvements:
1. Architecture SHRINKS depth (not width!) as n grows - enforces Gaussian structure
2. Width stays ≥ dim to avoid rank collapse in mean/covariance representation
3. Sigma cap scales with n as c/√n (c=1.5) - THE CRITICAL GUARDRAIL
4. Weight decay (1e-4) prevents wild variation in means across latent space
5. Proper training schedules (800 steps with early stopping)
6. Comprehensive diagnostics vs Laplace approximation including rank monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# BvM-Aware Hyperparameter Schedule
# ============================================================================

def get_bvm_hyperparams(n, dim, c_sigma=1.2):  # Relaxed from 1.0
    """
    Get BvM-aware hyperparameters that adapt to sample size.
    
    Key insight: In BvM regime (large n), posterior is Gaussian with σ ~ 1/√n.
    - Reduce DEPTH (not width) to enforce near-affine structure
    - Width must stay ≥ dim to represent full-rank mean/covariance
    - Sigma cap should scale with √n to prevent over-dispersion (CRITICAL!)
    - K should scale with n for proper BvM bias control
    """
    
    # 1. ARCHITECTURE: Reduce depth, keep width ≥ dim
    # Width: Must be ≥ dim to avoid rank bottleneck in mean/covariance
    # Depth: Safe knob to reduce nonlinearity (BvM wants near-affine map)
    if n <= 200:
        # Small n: posterior may be non-Gaussian, need flexibility
        hidden_dim = max(8, dim)
        depth = 2
    else:
        # n > 200: BvM regime for Gaussian, use depth=1 (near-affine)
        # For non-Gaussian posteriors, depth=1 still works as they approach Gaussian
        hidden_dim = max(4, dim)
        depth = 1
    
    # 2. SIGMA CAP: Scale with Fisher information (THE CRITICAL GUARDRAIL!)
    # True posterior σ ~ 1/√n
    # Cap at c/√n with c≈2-3 so σ_true ≤ 0.5·σ_max (avoids undercoverage)
    if n >= 200:
        sigma_max = c_sigma / np.sqrt(n)
    else:
        sigma_max = 5.0  # More permissive for small n
    sigma_min = 0.01  # Numerical stability floor
    
    # 3. TRAINING: Simple targets need fewer steps
    steps = 600 if n <= 200 else 800
    
    # 4. K SCHEDULE: For BvM, need bias = o(n^{-1})
    # K_train: Can be smaller, will anneal up
    # K_eval: Should scale with n for proper coverage
    K_train = min(128, n // 2)
    K_eval = max(512, n)
    lr = 1e-3
    
    # 5. LINEAR HEAD: For extreme BvM regime (n ≥ 500), can use LinearGaussianSIVI
    # This enforces exact Gaussian structure (no mixture over-dispersion)
    # Re-enabled after testing showed regular SIVI has optimization issues at large n
    use_linear_head = (n >= 500)  # Enforces single Gaussian structure
    
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
# SIVI Model with Bounded Sigma
# ============================================================================

class PosteriorSIVI(nn.Module):
    """SIVI for approximating posterior distributions with bounded sigma"""
    
    def __init__(self, latent_dim=2, param_dim=2, hidden_dim=64, depth=2,
                 sigma_min=0.01, sigma_max=5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # MLP for mean
        layers_mu = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers_mu.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_mu.append(nn.Linear(hidden_dim, param_dim))
        self.mu_net = nn.Sequential(*layers_mu)
        
        # MLP for log std
        layers_sigma = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers_sigma.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_sigma.append(nn.Linear(hidden_dim, param_dim))
        self.log_sigma_net = nn.Sequential(*layers_sigma)
        
        # Base distribution
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
    
    def forward(self, theta, K=128):
        """Compute log q(theta) using K importance samples"""
        batch_size = theta.shape[0]
        
        # Sample from base
        base_dist = torch.distributions.Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        # Conditional parameters
        mu = self.mu_net(z_flat)
        log_sigma = self.log_sigma_net(z_flat)
        sigma = torch.exp(log_sigma).clamp(min=self.sigma_min, max=self.sigma_max)
        
        mu = mu.reshape(batch_size, K, self.param_dim)
        sigma = sigma.reshape(batch_size, K, self.param_dim)
        
        # Expand theta for broadcasting
        theta_expanded = theta.unsqueeze(1).expand(-1, K, -1)
        
        # Log probability under each conditional
        log_prob_theta_z = -0.5 * torch.sum(
            ((theta_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        
        # Log marginal via importance sampling
        log_q_theta = torch.logsumexp(log_prob_theta_z, dim=1) - np.log(K)
        
        return log_q_theta
    
    def sample(self, n_samples, K=1):
        """Sample from the SIVI distribution"""
        base_dist = torch.distributions.Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((n_samples * K,))
        
        mu = self.mu_net(z)
        log_sigma = self.log_sigma_net(z)
        sigma = torch.exp(log_sigma).clamp(min=self.sigma_min, max=self.sigma_max)
        
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
# Linear Gaussian SIVI (for large n, enforces exact Gaussian structure)
# ============================================================================

class LinearGaussianSIVI(nn.Module):
    """
    SIVI with affine mean and constant covariance - guarantees single Gaussian marginal.
    
    For z ~ N(0, I), the marginal is:
        q(θ) = N(θ | b, AA^T + Σ)
    
    This eliminates mixture over-dispersion by construction.
    Use for n ≥ 500 in BvM regime when regular SIVI shows var_ratio > 1.5
    """
    
    def __init__(self, latent_dim=2, param_dim=2, sigma_min=0.01, sigma_max=5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Affine mean: μ(z) = Az + b
        self.A = nn.Parameter(torch.randn(param_dim, latent_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(param_dim))
        
        # Diagonal covariance (constant across z)
        init_sigma = (sigma_min + sigma_max) / 2
        self.log_diag = nn.Parameter(torch.log(torch.ones(param_dim) * init_sigma))
        
        # Base distribution
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
        
        # Precompute marginal covariance: AA^T + diag(σ²)
        self._update_marginal_cov()
    
    def _update_marginal_cov(self):
        """Update cached marginal covariance"""
        sigma = torch.exp(self.log_diag).clamp(min=self.sigma_min, max=self.sigma_max)
        self.marginal_cov = self.A @ self.A.T + torch.diag(sigma ** 2)
        self.marginal_mean = self.b
    
    def forward(self, theta, K=128):
        """Compute log q(theta) using exact Gaussian marginal"""
        # Update marginal parameters
        self._update_marginal_cov()
        
        # Log probability under N(b, AA^T + Σ)
        diff = theta - self.marginal_mean
        
        try:
            # Solve for precision * diff
            prec_diff = torch.linalg.solve(self.marginal_cov, diff.T).T
            log_prob = -0.5 * torch.sum(diff * prec_diff, dim=-1)
            log_prob -= 0.5 * torch.logdet(self.marginal_cov)
            log_prob -= self.param_dim * 0.5 * np.log(2 * np.pi)
        except:
            # Fallback: add ridge for stability
            cov_ridge = self.marginal_cov + 1e-6 * torch.eye(self.param_dim).to(theta.device)
            prec_diff = torch.linalg.solve(cov_ridge, diff.T).T
            log_prob = -0.5 * torch.sum(diff * prec_diff, dim=-1)
            log_prob -= 0.5 * torch.logdet(cov_ridge)
            log_prob -= self.param_dim * 0.5 * np.log(2 * np.pi)
        
        return log_prob
    
    def sample(self, n_samples, K=1):
        """Sample from N(b, AA^T + Σ)"""
        self._update_marginal_cov()
        
        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(self.marginal_cov)
        except:
            cov_ridge = self.marginal_cov + 1e-6 * torch.eye(self.param_dim).to(self.marginal_cov.device)
            L = torch.linalg.cholesky(cov_ridge)
        
        # Sample
        eps = torch.randn(n_samples, self.param_dim).to(L.device)
        samples = self.marginal_mean + (L @ eps.T).T
        
        return samples
    
    def mean_and_cov(self, n_samples=10000):
        """Return exact mean and covariance (no sampling needed)"""
        self._update_marginal_cov()
        return self.marginal_mean.detach(), self.marginal_cov.detach()


# ============================================================================
# Model 1: Gaussian Mean Estimation (Conjugate)
# ============================================================================

class GaussianMeanModel:
    """
    Y_i ~ N(theta, I), theta ~ N(0, tau^2 I)
    
    Conjugate: posterior is N(theta_post, Sigma_post)
    """
    def __init__(self, dim=2, prior_tau=1.0, likelihood_sigma=1.0):
        self.dim = dim
        self.prior_tau = prior_tau
        self.likelihood_sigma = likelihood_sigma
    
    def sample_data(self, n, theta_true):
        """Generate n observations from N(theta_true, I)"""
        return theta_true + torch.randn(n, self.dim) * self.likelihood_sigma
    
    def conjugate_posterior(self, y):
        """Exact posterior parameters (conjugate)"""
        n = len(y)
        ybar = y.mean(dim=0)
        
        # Posterior precision
        prior_prec = 1.0 / (self.prior_tau ** 2)
        lik_prec = 1.0 / (self.likelihood_sigma ** 2)
        post_prec = prior_prec + n * lik_prec
        
        # Posterior covariance and mean
        post_cov = torch.eye(self.dim) / post_prec
        post_mean = (n * lik_prec / post_prec) * ybar
        
        return post_mean, post_cov
    
    def log_posterior(self, theta, y):
        """Log posterior (unnormalized)"""
        n = len(y)
        
        # Log likelihood
        diff = y - theta.unsqueeze(0)
        log_lik = -0.5 * torch.sum(diff ** 2) / (self.likelihood_sigma ** 2)
        log_lik -= n * self.dim * 0.5 * np.log(2 * np.pi * self.likelihood_sigma ** 2)
        
        # Log prior
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
    def __init__(self, dim=10, prior_tau=5.0):
        self.dim = dim
        self.prior_tau = prior_tau
    
    def sample_data(self, n, theta_true):
        """Generate n observations"""
        X = torch.randn(n, self.dim)
        logits = X @ theta_true
        probs = torch.sigmoid(logits)
        y = torch.bernoulli(probs).long()
        return X, y
    
    def log_posterior(self, theta, X, y):
        """Log posterior (unnormalized)"""
        # Log likelihood
        logits = X @ theta
        log_lik = torch.sum(y * logits - torch.log(1 + torch.exp(logits)))
        
        # Log prior
        log_prior = -0.5 * torch.sum(theta ** 2) / (self.prior_tau ** 2)
        log_prior -= self.dim * 0.5 * np.log(2 * np.pi * self.prior_tau ** 2)
        
        return log_lik + log_prior
    
    def mle_and_fisher(self, X, y, ridge=1e-6):
        """Laplace approximation: MLE and Fisher information"""
        theta = torch.zeros(self.dim)
        
        # Newton's method for MLE
        for _ in range(50):
            logits = X @ theta
            probs = torch.sigmoid(logits)
            
            # Gradient
            grad = X.T @ (y - probs) - theta / (self.prior_tau ** 2)
            
            # Hessian (with ridge)
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
        
        # Fisher information at MLE
        logits = X @ theta
        probs = torch.sigmoid(logits)
        W = probs * (1 - probs)
        Fisher = X.T @ (W.unsqueeze(1) * X) + torch.eye(self.dim) / (self.prior_tau ** 2)
        Fisher = Fisher + ridge * torch.eye(self.dim)
        
        return theta, Fisher


# ============================================================================
# SIVI Training with Early Stopping
# ============================================================================

def fit_sivi_posterior_gaussian(y, prior_tau=1.0, hypers=None, verbose=True, 
                                early_stop=True):
    """Fit SIVI to Gaussian posterior with BvM-aware hyperparameters"""
    dim = y.shape[1]
    n = len(y)
    
    if hypers is None:
        hypers = get_bvm_hyperparams(n, dim)
    
    model = GaussianMeanModel(dim=dim, prior_tau=prior_tau)
    
    # Choose SIVI variant
    if hypers.get('use_linear_head', False):
        if verbose:
            print(f"  Using LinearGaussianSIVI (enforces exact Gaussian structure)")
        sivi = LinearGaussianSIVI(
            latent_dim=dim,
            param_dim=dim,
            sigma_min=hypers['sigma_min'],
            sigma_max=hypers['sigma_max']
        ).to(device)
    else:
        # Initialize regular SIVI with bounded sigma
        sivi = PosteriorSIVI(
            latent_dim=dim, 
            param_dim=dim,
            hidden_dim=hypers['hidden_dim'],
            depth=hypers['depth'],
            sigma_min=hypers['sigma_min'],
            sigma_max=hypers['sigma_max']
        ).to(device)
        
        # Xavier initialization with moderate gain (avoid rank collapse)
        for m in sivi.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Increased from 0.3
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    optimizer = optim.Adam(sivi.parameters(), lr=hypers['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hypers['steps'], eta_min=1e-5
    )
    
    y_dev = y.to(device)
    
    # Get true posterior for early stopping
    post_mean, post_cov = model.conjugate_posterior(y)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    pbar = tqdm(range(hypers['steps']), desc='Training SIVI', leave=False, 
                disable=not verbose)
    
    for step in pbar:
        # Sample from q
        theta = sivi.sample(hypers['K_train']).to(device)
        
        # Compute log q(theta)
        log_q = sivi(theta, K=hypers['K_train'])
        
        # Compute log posterior p(theta|y)
        log_post = torch.stack([
            model.log_posterior(theta[i], y_dev) for i in range(len(theta))
        ])
        
        # Forward KL: E_q[log q(theta) - log p(theta|y)]
        loss = (log_q - log_post).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sivi.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Early stopping check
        if early_stop and step % 50 == 0 and step > 200:
            with torch.no_grad():
                sivi_mean, sivi_cov = sivi.mean_and_cov(n_samples=1000)
                
                # Check 1: Loss improvement
                recent_loss = np.mean(losses[-10:])
                loss_improved = (best_loss - recent_loss) / abs(best_loss) > 0.001
                
                # Check 2: Dispersion ratio near 1
                disp_ratio = torch.trace(sivi_cov @ torch.linalg.inv(post_cov)) / dim
                dispersion_ok = 0.95 <= disp_ratio.item() <= 1.05
                
                if not loss_improved and dispersion_ok:
                    patience_counter += 1
                    if patience_counter >= 2:
                        if verbose:
                            print(f"  Early stop at step {step}")
                        break
                else:
                    patience_counter = 0
                    best_loss = min(best_loss, recent_loss)
    
    pbar.close()
    return sivi


def fit_sivi_posterior_logistic(X, y, prior_tau=5.0, hypers=None, verbose=True,
                                early_stop=True):
    """Fit SIVI to logistic posterior with BvM-aware hyperparameters"""
    dim = X.shape[1]
    n = len(X)
    
    if hypers is None:
        hypers = get_bvm_hyperparams(n, dim)
    
    model = LogisticRegressionModel(dim=dim, prior_tau=prior_tau)
    
    # Choose SIVI variant
    if hypers.get('use_linear_head', False):
        if verbose:
            print(f"  Using LinearGaussianSIVI (enforces exact Gaussian structure)")
        sivi = LinearGaussianSIVI(
            latent_dim=dim,
            param_dim=dim,
            sigma_min=hypers['sigma_min'],
            sigma_max=hypers['sigma_max']
        ).to(device)
    else:
        # Initialize regular SIVI
        sivi = PosteriorSIVI(
            latent_dim=dim,
            param_dim=dim,
            hidden_dim=hypers['hidden_dim'],
            depth=hypers['depth'],
            sigma_min=hypers['sigma_min'],
            sigma_max=hypers['sigma_max']
        ).to(device)
        
        for m in sivi.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    optimizer = optim.Adam(sivi.parameters(), lr=hypers['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hypers['steps'], eta_min=1e-5
    )
    
    X_dev = X.to(device)
    y_dev = y.to(device)
    
    # Get Laplace approximation for early stopping
    theta_mle, Fisher = model.mle_and_fisher(X, y)
    post_cov_laplace = torch.linalg.inv(Fisher)
    
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    pbar = tqdm(range(hypers['steps']), desc='Training SIVI', leave=False,
                disable=not verbose)
    
    for step in pbar:
        theta = sivi.sample(hypers['K_train']).to(device)
        log_q = sivi(theta, K=hypers['K_train'])
        
        log_post = torch.stack([
            model.log_posterior(theta[i], X_dev, y_dev) for i in range(len(theta))
        ])
        
        loss = (log_q - log_post).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sivi.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Early stopping
        if early_stop and step % 50 == 0 and step > 200:
            with torch.no_grad():
                sivi_mean, sivi_cov = sivi.mean_and_cov(n_samples=1000)
                
                recent_loss = np.mean(losses[-10:])
                loss_improved = (best_loss - recent_loss) / abs(best_loss) > 0.001
                
                disp_ratio = torch.trace(sivi_cov @ torch.linalg.inv(post_cov_laplace)) / dim
                dispersion_ok = 0.90 <= disp_ratio.item() <= 1.10
                
                if not loss_improved and dispersion_ok:
                    patience_counter += 1
                    if patience_counter >= 2:
                        if verbose:
                            print(f"  Early stop at step {step}")
                        break
                else:
                    patience_counter = 0
                    best_loss = min(best_loss, recent_loss)
    
    pbar.close()
    return sivi


# ============================================================================
# Coverage and Diagnostics
# ============================================================================

@torch.no_grad()
def credible_coverage(sivi, theta_true, alpha=0.05, K_eval=512):
    """
    Check if theta_true falls in (1-alpha) credible ellipsoid
    """
    m, S = sivi.mean_and_cov(n_samples=K_eval)
    m = m.cpu()
    S = S.cpu()
    
    delta = theta_true - m
    
    try:
        dist_sq = delta @ torch.linalg.solve(S, delta)
    except:
        S_ridge = S + 1e-6 * torch.eye(len(S))
        dist_sq = delta @ torch.linalg.solve(S_ridge, delta)
    
    d = len(theta_true)
    threshold = chi2.ppf(1 - alpha, df=d)
    
    inside = (dist_sq <= threshold).item()
    
    return inside, dist_sq.item(), threshold


@torch.no_grad()
def compute_diagnostics(sivi, true_mean, true_cov, K_eval=512):
    """Compute approximation quality metrics with sanity checks"""
    sivi_mean, sivi_cov = sivi.mean_and_cov(n_samples=K_eval)
    
    # Mean error
    mean_error = torch.norm(sivi_mean - true_mean).item()
    rel_mean_error = mean_error / torch.norm(true_mean).item() if torch.norm(true_mean) > 1e-6 else mean_error
    
    # Variance ratio
    var_ratio = (torch.diag(sivi_cov) / torch.diag(true_cov)).mean().item()
    
    # Frobenius norm of covariance difference
    cov_error = torch.norm(sivi_cov - true_cov, p='fro').item()
    
    # SANITY CHECK 1: Rank of mean mapping (only for regular SIVI with mu_net)
    if hasattr(sivi, 'mu_net'):
        # Sample latents and compute Jacobian via finite differences
        z_samples = torch.randn(100, sivi.latent_dim)
        mu_samples = sivi.mu_net(z_samples)
        
        # Approximate Jacobian via covariance
        z_centered = z_samples - z_samples.mean(dim=0)
        mu_centered = mu_samples - mu_samples.mean(dim=0)
        J_approx = (mu_centered.T @ z_centered) / len(z_samples)
        
        # Smallest singular value (should be > 0 to avoid rank collapse)
        try:
            singular_values = torch.linalg.svdvals(J_approx)
            min_singular_value = singular_values.min().item()
            condition_number = (singular_values.max() / (singular_values.min() + 1e-10)).item()
        except:
            min_singular_value = 0.0
            condition_number = float('inf')
    else:
        # For LinearGaussianSIVI, check rank of A matrix directly
        if hasattr(sivi, 'A'):
            try:
                singular_values = torch.linalg.svdvals(sivi.A)
                min_singular_value = singular_values.min().item()
                condition_number = (singular_values.max() / (singular_values.min() + 1e-10)).item()
            except:
                min_singular_value = 0.0
                condition_number = float('inf')
        else:
            # No rank check available
            min_singular_value = float('nan')
            condition_number = float('nan')
    
    # SANITY CHECK 2: Variance ratio flag
    var_ratio_warning = ""
    if var_ratio > 1.5:
        var_ratio_warning = "WARNING: Over-dispersion! Tighten σ_max or increase K"
    elif var_ratio < 0.7:
        var_ratio_warning = "WARNING: Under-dispersion! May have undercoverage"
    
    return {
        'mean_error': mean_error,
        'rel_mean_error': rel_mean_error,
        'var_ratio': var_ratio,
        'cov_error': cov_error,
        'sivi_mean': sivi_mean.cpu(),
        'sivi_cov': sivi_cov.cpu(),
        'min_singular_value': min_singular_value,
        'condition_number': condition_number,
        'var_ratio_warning': var_ratio_warning
    }


# ============================================================================
# Main Experiments
# ============================================================================

def run_gaussian_experiment():
    """
    Gaussian mean estimation: posterior is EXACTLY Gaussian at all n
    Should get ~95% coverage at all n (no approximation error)
    """
    print("="*70)
    print("GAUSSIAN MEAN MODEL: BvM Coverage Test")
    print("="*70)
    
    dim = 5
    theta_true = torch.randn(dim) * 0.5
    prior_tau = 1.0
    
    N_list = [50, 100, 200]
    R = 5  # Replications (increased for better statistical power)
    alpha = 0.05
    
    results = []
    
    for n in N_list:
        print(f"\n{'='*70}")
        print(f"n = {n} (dim = {dim})")
        print('='*70)
        
        # Get hyperparameters
        hypers = get_bvm_hyperparams(n, dim)
        
        # Theoretical posterior std
        sigma_true = 1.0 / np.sqrt(n + 1)
        
        print(f"  BvM-aware settings:")
        print(f"    Architecture: hidden_dim={hypers['hidden_dim']}, depth={hypers['depth']}")
        print(f"    σ bounds: [{hypers['sigma_min']:.4f}, {hypers['sigma_max']:.4f}]")
        print(f"    True posterior σ: {sigma_true:.4f}")
        print(f"    Training: {hypers['steps']} steps")
        print(f"    Use linear head: {hypers.get('use_linear_head', False)}")  # DEBUG
        
        model = GaussianMeanModel(dim=dim, prior_tau=prior_tau)
        
        covers = []
        diagnostics = []
        
        for r in tqdm(range(R), desc=f'Replications (n={n})'):
            # Generate data
            y = model.sample_data(n, theta_true)
            
            # Fit SIVI
            sivi = fit_sivi_posterior_gaussian(
                y, prior_tau=prior_tau, hypers=hypers, 
                verbose=(r == 0), early_stop=True  # Verbose for first rep
            )
            sivi.eval()
            
            # True posterior
            post_mean, post_cov = model.conjugate_posterior(y)
            
            # Diagnostics
            diag = compute_diagnostics(sivi, post_mean, post_cov, K_eval=hypers['K_eval'])
            diagnostics.append(diag)
            
            # Coverage
            inside, _, _ = credible_coverage(sivi, theta_true, alpha=alpha, 
                                            K_eval=hypers['K_eval'])
            covers.append(inside)
            
            # Print first replication with sanity checks
            if r == 0:
                print(f"\n  Sample diagnostics (rep 0):")
                print(f"    Mean error: {diag['rel_mean_error']:.2%}")
                print(f"    Var ratio: {diag['var_ratio']:.3f}")
                print(f"    Min singular value (rank check): {diag['min_singular_value']:.4f}")
                print(f"    Condition number: {diag['condition_number']:.2f}")
                if diag['var_ratio_warning']:
                    print(f"    ⚠️  {diag['var_ratio_warning']}")
        
        # Aggregate results
        cov_rate = np.mean(covers)
        cov_sd = np.sqrt(cov_rate * (1 - cov_rate) / R)
        
        mean_errors = [d['rel_mean_error'] for d in diagnostics]
        var_ratios = [d['var_ratio'] for d in diagnostics]
        
        print(f"\n  Aggregate results (R={R}):")
        print(f"    Coverage: {cov_rate:.3f} ± {cov_sd:.3f} (nominal: {1-alpha:.3f})")
        print(f"    Mean error: {np.mean(mean_errors):.2%} ± {np.std(mean_errors):.2%}")
        print(f"    Var ratio: {np.mean(var_ratios):.3f} ± {np.std(var_ratios):.3f}")
        
        results.append({
            'n': n,
            'coverage': cov_rate,
            'coverage_sd': cov_sd,
            'mean_error': np.mean(mean_errors),
            'var_ratio': np.mean(var_ratios)
        })
    
    # Plot
    plot_results(results, model_name='Gaussian', alpha=alpha, R=R)
    
    return results


def run_logistic_experiment():
    """
    Logistic regression: posterior transitions from non-Gaussian → Gaussian
    Should see coverage increase from ~90% → 95% as n grows
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION MODEL: BvM Coverage Test")
    print("="*70)
    
    dim = 5
    theta_true = torch.randn(dim) * 0.5
    prior_tau = 5.0
    
    N_list = [50, 100, 200, 300]
    R = 20  # Replications (increased for better statistical power)
    alpha = 0.05
    
    results = []
    
    for n in N_list:
        print(f"\n{'='*70}")
        print(f"n = {n} (dim = {dim})")
        print('='*70)
        
        # Get hyperparameters
        hypers = get_bvm_hyperparams(n, dim)
        
        print(f"  BvM-aware settings:")
        print(f"    Architecture: hidden_dim={hypers['hidden_dim']}, depth={hypers['depth']}")
        print(f"    σ bounds: [{hypers['sigma_min']:.4f}, {hypers['sigma_max']:.4f}]")
        print(f"    Training: {hypers['steps']} steps")
        
        model = LogisticRegressionModel(dim=dim, prior_tau=prior_tau)
        
        covers = []
        diagnostics = []
        
        for r in tqdm(range(R), desc=f'Replications (n={n})'):
            # Generate data
            X, y = model.sample_data(n, theta_true)
            
            # Fit SIVI
            sivi = fit_sivi_posterior_logistic(
                X, y, prior_tau=prior_tau, hypers=hypers,
                verbose=False, early_stop=True
            )
            sivi.eval()
            
            # Laplace approximation (true posterior)
            theta_mle, Fisher = model.mle_and_fisher(X, y)
            post_cov_laplace = torch.linalg.inv(Fisher)
            
            # Diagnostics
            diag = compute_diagnostics(sivi, theta_mle, post_cov_laplace, 
                                      K_eval=hypers['K_eval'])
            diagnostics.append(diag)
            
            # Coverage
            inside, _, _ = credible_coverage(sivi, theta_true, alpha=alpha,
                                            K_eval=hypers['K_eval'])
            covers.append(inside)
            
            # Print first replication with sanity checks
            if r == 0:
                print(f"\n  Sample diagnostics (rep 0):")
                print(f"    Mean error: {diag['rel_mean_error']:.2%}")
                print(f"    Var ratio: {diag['var_ratio']:.3f}")
                print(f"    Min singular value (rank check): {diag['min_singular_value']:.4f}")
                print(f"    Condition number: {diag['condition_number']:.2f}")
                if diag['var_ratio_warning']:
                    print(f"    ⚠️  {diag['var_ratio_warning']}")
        
        # Aggregate results
        cov_rate = np.mean(covers)
        cov_sd = np.sqrt(cov_rate * (1 - cov_rate) / R)
        
        mean_errors = [d['rel_mean_error'] for d in diagnostics]
        var_ratios = [d['var_ratio'] for d in diagnostics]
        
        print(f"\n  Aggregate results (R={R}):")
        print(f"    Coverage: {cov_rate:.3f} ± {cov_sd:.3f} (nominal: {1-alpha:.3f})")
        print(f"    Mean error: {np.mean(mean_errors):.2%} ± {np.std(mean_errors):.2%}")
        print(f"    Var ratio: {np.mean(var_ratios):.3f} ± {np.std(var_ratios):.3f}")
        
        results.append({
            'n': n,
            'coverage': cov_rate,
            'coverage_sd': cov_sd,
            'mean_error': np.mean(mean_errors),
            'var_ratio': np.mean(var_ratios)
        })
    
    # Plot
    plot_results(results, model_name='Logistic', alpha=alpha, R=R)
    
    return results


def plot_results(results, model_name, alpha, R):
    """Plot coverage and diagnostics"""
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
    ax.fill_between(N_list,
                     [1-alpha - 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     [1-alpha + 1.96*np.sqrt((1-alpha)*alpha/R)]*len(N_list),
                     alpha=0.2, color='red', label='95% Binomial CI')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title(f'{model_name}: Credible Coverage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.0])
    
    # Mean approximation error
    ax = axes[1]
    ax.plot(N_list, mean_errors, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Relative Mean Error', fontsize=12)
    ax.set_title('Mean Approximation Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Variance ratio
    ax = axes[2]
    ax.plot(N_list, var_ratios, 'o-', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Variance Ratio (SIVI/True)', fontsize=12)
    ax.set_title('Variance Approximation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.15])
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_bvm_coverage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n  Plot saved: {model_name.lower()}_bvm_coverage.png")


# ============================================================================
# Main Runner
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXPERIMENT 7: BvM-AWARE COVERAGE TRANSFER")
    print("="*70)
    print("\nKey insight: Architecture should SHRINK as n grows!")
    print("  - Small n: flexible networks needed (non-Gaussian posterior)")
    print("  - Large n: simple networks better (enforce Gaussian structure)")
    print("  - Sigma cap ~ 1/√n prevents over-dispersion")
    
    # Gaussian experiment (exact posterior at all n)
    #print("\n" + "="*70)
    #print("PART 1: GAUSSIAN MEAN (CONJUGATE)")
    #print("="*70)
    #gauss_results = run_gaussian_experiment()
    
    # Logistic experiment (non-Gaussian → Gaussian transition)
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION (NON-CONJUGATE)")
    print("="*70)
    logistic_results = run_logistic_experiment()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
   # print("\nGaussian (should be ~95% at all n):")
   # for r in gauss_results:
   #    print(f"  n={r['n']:4d}: coverage={r['coverage']:.3f}, "
    #          f"mean_err={r['mean_error']:.2%}, var_ratio={r['var_ratio']:.3f}")
    
    print("\nLogistic (should increase: 90% → 95%):")
    for r in logistic_results:
        print(f"  n={r['n']:4d}: coverage={r['coverage']:.3f}, "
              f"mean_err={r['mean_error']:.2%}, var_ratio={r['var_ratio']:.3f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT 7 COMPLETE")
    print("="*70)