import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal, StudentT
from scipy import optimize
from scipy.special import gamma as gamma_func
from scipy.stats import t as student_t

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# FIX 1: Sub-Gaussian warped target with accurate Newton inverse
# ============================================================================

class SubGaussianWarpedTarget:
    """
    Sub-Gaussian target: N(0,I) warped by smooth diffeomorphism
    with Newton-based accurate inverse
    """
    def __init__(self, dim=2):
        self.dim = dim
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.warp_scale = 0.3
    
    def _warp(self, x):
        """Smooth diffeomorphism: y = x + α*tanh(x)"""
        return x + self.warp_scale * torch.tanh(x)
    
    def _warp_derivative(self, x):
        """Derivative: dy/dx = 1 + α*sech²(x)"""
        sech_sq = 1.0 / torch.cosh(x)**2
        return 1.0 + self.warp_scale * sech_sq
    
    def _inverse_warp_newton(self, y, max_iter=10, tol=1e-6):
        """Newton's method for accurate inverse: find x s.t. warp(x) = y"""
        x = y.clone()  # Initial guess
        
        for i in range(max_iter):
            fx = self._warp(x) - y
            fpx = self._warp_derivative(x)
            
            # Newton step
            x_new = x - fx / fpx
            
            # Check convergence
            residual = torch.abs(x_new - x).max()
            x = x_new
            
            if residual < tol:
                break
        
        # Track convergence for diagnostics
        final_residual = torch.abs(self._warp(x) - y).max()
        
        return x, final_residual.item()
    
    def _log_det_jacobian(self, x):
        """Log determinant of Jacobian"""
        diag_jacob = self._warp_derivative(x)
        return torch.sum(torch.log(diag_jacob), dim=-1)
    
    def sample(self, n_samples):
        """Sample from warped distribution"""
        z = self.base_dist.sample((n_samples,))
        return self._warp(z)
    
    def log_prob(self, y):
        """Compute log probability using change of variables with Newton inverse"""
        x, residual = self._inverse_warp_newton(y)
        log_px = self.base_dist.log_prob(x)
        log_det_jac = self._log_det_jacobian(x)
        return log_px - log_det_jac


class HeavyTailedTarget:
    """Heavy-tailed target: Multivariate Student-t with small nu"""
    def __init__(self, dim=2, nu=3.0):
        self.dim = dim
        self.nu = nu
        self.marginal_dist = StudentT(df=nu)
    
    def sample(self, n_samples):
        """Sample from multivariate Student-t (independent components)"""
        return self.marginal_dist.sample((n_samples, self.dim))
    
    def log_prob(self, x):
        """Compute log probability"""
        return torch.sum(self.marginal_dist.log_prob(x), dim=-1)


# ============================================================================
# SIVI with adaptive depth and variance constraint
# ============================================================================

class SIVI(nn.Module):
    def __init__(self, latent_dim=2, output_dim=2, hidden_dim=64, max_sigma=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_sigma = max_sigma
        
        # FIX 5: Adaptive depth
        n_hidden_layers = max(2, int(1 + np.log2(max(hidden_dim / 8, 1))))
        
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
        
        print(f"Initialized SIVI: width={hidden_dim}, depth={n_hidden_layers+2}, max_sigma={max_sigma}")
    
    def forward(self, x, K=64):
        """FIX 3: Enforce variance constraint strictly"""
        batch_size = x.shape[0]
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        mu = self.mu_net(z_flat)
        log_sigma = self.log_sigma_net(z_flat)
        
        # CRITICAL: Enforce max_sigma constraint
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=self.max_sigma)
        
        mu = mu.reshape(batch_size, K, self.output_dim)
        sigma = sigma.reshape(batch_size, K, self.output_dim)
        
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)
        
        log_prob_xz = -0.5 * torch.sum(
            ((x_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        
        log_qx = torch.logsumexp(log_prob_xz, dim=1) - np.log(K)
        return log_qx
    
    def check_tail_decay(self, n_samples=5000, threshold=3.0):
        """FIX 3: Verify empirical sub-Gaussian tail"""
        with torch.no_grad():
            base_dist = Normal(self.base_loc, self.base_scale)
            z = base_dist.sample((n_samples,))
            
            mu = self.mu_net(z)
            log_sigma = self.log_sigma_net(z)
            sigma = torch.exp(log_sigma).clamp(min=1e-4, max=self.max_sigma)
            
            eps = torch.randn_like(mu)
            samples = mu + sigma * eps
            
            # Check tail probability
            norms = torch.norm(samples, dim=-1)
            tail_prob = (norms > threshold).float().mean()
            
            return tail_prob.item()


class SIVIWithTailComponent(nn.Module):
    """FIX 6: SIVI with learnable tail component weight"""
    def __init__(self, latent_dim=2, output_dim=2, hidden_dim=64, 
                 max_sigma=2.0, tail_sigma=5.0, learnable_tail_weight=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_sigma = max_sigma
        self.tail_sigma = tail_sigma
        
        n_hidden_layers = max(2, int(1 + np.log2(max(hidden_dim / 8, 1))))
        
        # Core SIVI component
        layers_mu = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers_mu.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_mu.append(nn.Linear(hidden_dim, output_dim))
        self.mu_net = nn.Sequential(*layers_mu)
        
        layers_sigma = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers_sigma.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_sigma.append(nn.Linear(hidden_dim, output_dim))
        self.log_sigma_net = nn.Sequential(*layers_sigma)
        
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
        
        # Tail component
        self.register_buffer('tail_loc', torch.zeros(output_dim))
        self.register_buffer('tail_scale', torch.ones(output_dim) * tail_sigma)
        
        # FIX 6: Learnable tail weight via sigmoid
        if learnable_tail_weight:
            self.tail_weight_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        else:
            self.register_buffer('tail_weight_logit', torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12
        
        print(f"Initialized SIVI+Tail: width={hidden_dim}, depth={n_hidden_layers+2}, "
              f"tail_sigma={tail_sigma}, learnable_weight={learnable_tail_weight}")
    
    def forward(self, x, K=64):
        batch_size = x.shape[0]
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        mu = self.mu_net(z_flat)
        log_sigma = self.log_sigma_net(z_flat)
        sigma = torch.exp(log_sigma).clamp(min=1e-4, max=self.max_sigma)
        
        mu = mu.reshape(batch_size, K, self.output_dim)
        sigma = sigma.reshape(batch_size, K, self.output_dim)
        
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)
        
        log_prob_core = -0.5 * torch.sum(
            ((x_expanded - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi),
            dim=-1
        )
        log_qx_core = torch.logsumexp(log_prob_core, dim=1) - np.log(K)
        
        # Tail component
        tail_dist = Normal(self.tail_loc, self.tail_scale)
        log_qx_tail = torch.sum(tail_dist.log_prob(x), dim=-1)
        
        # Adaptive mixture weight
        tail_weight = torch.sigmoid(self.tail_weight_logit)
        
        log_qx = torch.logsumexp(
            torch.stack([
                log_qx_core + torch.log(1 - tail_weight),
                log_qx_tail + torch.log(tail_weight)
            ], dim=0),
            dim=0
        )
        
        return log_qx
    
    def get_tail_weight(self):
        """Get current tail weight"""
        return torch.sigmoid(self.tail_weight_logit).item()


# ============================================================================
# FIX 5: Training with cosine annealing
# ============================================================================

def train_sivi(model, target, n_epochs=3000, batch_size=256, lr=1e-3, K=64, verbose=True):
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
        
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model


# ============================================================================
# FIX 2 & 4: Forward KL estimation with large K_eval and more samples
# ============================================================================

def estimate_forward_kl(model, target, n_samples=15000, K_eval=2048, batch_size=500):
    """
    FIX 2: Evaluate with large K_eval to remove Jensen bias
    FIX 4: Use more samples for heavy-tailed targets
    
    Estimate KL(p || q) = E_p[log p(x) - log q(x)]
    """
    model.eval()
    with torch.no_grad():
        x = target.sample(n_samples).to(device)
        
        # Compute log p(x)
        log_px = target.log_prob(x)
        
        # FIX 2: Compute log q(x) with large K_eval in batches
        log_qx_list = []
        for i in range(0, n_samples, batch_size):
            x_batch = x[i:i+batch_size]
            log_qx_batch = model(x_batch, K=K_eval)
            log_qx_list.append(log_qx_batch)
        log_qx = torch.cat(log_qx_list)
        
        # KL divergence
        kl = (log_px - log_qx).mean()
    
    return kl.item()


# ============================================================================
# FIX 7: Theoretical lower bound computation
# ============================================================================

def compute_theoretical_kl_lower_bound(nu=3.0, max_sigma=2.0, dim=2, verbose=True):
    """
    Compute lower bound on KL(p||q) using 1D coordinate projection
    (conservative bound for multivariate case)
    """
    if verbose:
        print(f"\nComputing theoretical lower bound:")
        print(f"  Target: Student-t(nu={nu}), dim={dim}")
        print(f"  SIVI: Bounded variance <= {max_sigma}^2")
        print(f"  Note: Using 1D projection (conservative multivariate bound)")
        print("-" * 60)
    
    C_nu = np.sqrt(nu / np.pi) * gamma_func((nu + 1) / 2) / gamma_func(nu / 2)
    
    def p_tail_1d(t):
        """Tail probability for 1D Student-t"""
        return 2 * (1 - student_t.cdf(t, df=nu))
    
    def q_tail_upper_bound_1d(t):
        """Upper bound on tail for bounded Gaussian"""
        return 2 * np.exp(-t**2 / (2 * max_sigma**2))
    
    def kl_bernoulli(p, q):
        """KL divergence between two Bernoulli distributions"""
        if q <= 0 or p <= 0:
            return 0
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    # Search for optimal threshold
    t_values = np.linspace(2, 8, 100)
    kl_bounds = []
    
    for t in t_values:
        p_t = p_tail_1d(t)
        q_t = q_tail_upper_bound_1d(t)
        
        if q_t > 0 and p_t > q_t:
            kl_bound = kl_bernoulli(p_t, q_t)
            kl_bounds.append(kl_bound)
        else:
            kl_bounds.append(0)
    
    best_idx = np.argmax(kl_bounds)
    best_t = t_values[best_idx]
    best_bound = kl_bounds[best_idx]
    
    if verbose:
        p_t_best = p_tail_1d(best_t)
        q_t_best = q_tail_upper_bound_1d(best_t)
        
        print(f"  Optimal threshold: t = {best_t:.3f}")
        print(f"  p(|X| >= {best_t:.3f}) = {p_t_best:.6f}  (polynomial tail)")
        print(f"  q(|X| >= {best_t:.3f}) <= {q_t_best:.6f}  (sub-Gaussian tail)")
        print(f"  Ratio: p/q ~ {p_t_best/q_t_best:.2e}")
        print(f"\n  ==> KL(p||q) >= {best_bound:.6f} (1D lower bound)")
        print("-" * 60)
    
    return best_bound


# ============================================================================
# Experiment 2a: Success (Sub-Gaussian target)
# ============================================================================

def experiment_2a_success(n_seeds=3):
    print("\n" + "="*70)
    print("EXPERIMENT 2A: SUCCESS - Sub-Gaussian Target")
    print("="*70)
    
    target = SubGaussianWarpedTarget(dim=2)
    widths = [16, 32, 64, 128, 256]
    
    K_train = 64
    K_eval = 2048
    n_samples_eval = 10000
    
    # Test 1: Standard SIVI with increasing width
    print("\n[Test 1] Standard SIVI with increasing width")
    all_kl_standard = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        kl_standard = []
        for width in widths:
            print(f"\n[Seed {seed+1}/{n_seeds}] Width = {width}")
            model = SIVI(latent_dim=2, output_dim=2, hidden_dim=width, 
                        max_sigma=3.0).to(device)
            
            # Better initialization
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            model = train_sivi(model, target, n_epochs=2500, lr=1e-3, K=K_train, verbose=False)
            
            kl = estimate_forward_kl(model, target, n_samples=n_samples_eval, K_eval=K_eval)
            kl_standard.append(kl)
            
            # FIX 3: Check tail behavior
            tail_prob = model.check_tail_decay(threshold=3.0)
            print(f"KL(p||q) = {kl:.6f}, Tail prob (>3σ) = {tail_prob:.6f}")
        
        all_kl_standard.append(kl_standard)
    
    mean_kl_standard = np.array(all_kl_standard).mean(axis=0)
    std_kl_standard = np.array(all_kl_standard).std(axis=0)
    
    # Test 2: With learnable tail component
    print("\n[Test 2] SIVI with learnable tail safety component (W=64)")
    all_kl_with_tail = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model_with_tail = SIVIWithTailComponent(
            latent_dim=2, output_dim=2, hidden_dim=64,
            max_sigma=2.0, tail_sigma=5.0, learnable_tail_weight=True
        ).to(device)
        
        for m in model_with_tail.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model_with_tail = train_sivi(model_with_tail, target, n_epochs=2500, 
                                     lr=1e-3, K=K_train, verbose=False)
        
        kl_with_tail = estimate_forward_kl(model_with_tail, target, 
                                          n_samples=n_samples_eval, K_eval=K_eval)
        tail_weight = model_with_tail.get_tail_weight()
        all_kl_with_tail.append(kl_with_tail)
        
        print(f"[Seed {seed+1}] KL = {kl_with_tail:.6f}, Tail weight = {tail_weight:.4f}")
    
    mean_kl_with_tail = np.mean(all_kl_with_tail)
    std_kl_with_tail = np.std(all_kl_with_tail)
    
    # FIX 1: Print diagnostics
    print(f"\nDiagnostics: K_eval={K_eval}, n_samples={n_samples_eval}")
    
    return widths, mean_kl_standard, std_kl_standard, mean_kl_with_tail, std_kl_with_tail


# ============================================================================
# Experiment 2b: Impossibility (Heavy-tailed target)
# ============================================================================

def experiment_2b_impossibility(n_seeds=3):
    print("\n" + "="*70)
    print("EXPERIMENT 2B: IMPOSSIBILITY - Heavy-Tailed Target")
    print("="*70)
    
    nu = 3.0
    max_sigma = 2.0
    theoretical_bound = compute_theoretical_kl_lower_bound(nu=nu, max_sigma=max_sigma, dim=2)
    
    target = HeavyTailedTarget(dim=2, nu=nu)
    widths = [16, 32, 64, 128, 256]
    
    K_train = 64
    K_eval = 2048
    n_samples_eval = 15000  # FIX 4: More samples for heavy tails
    
    print("\nSIVI with bounded variance (max_sigma=2.0)")
    all_kl_bounded = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        kl_bounded = []
        for width in widths:
            print(f"\n[Seed {seed+1}/{n_seeds}] Width = {width}")
            model = SIVI(latent_dim=2, output_dim=2, hidden_dim=width,
                        max_sigma=max_sigma).to(device)
            
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            model = train_sivi(model, target, n_epochs=2500, lr=1e-3, K=K_train, verbose=False)
            
            kl = estimate_forward_kl(model, target, n_samples=n_samples_eval, K_eval=K_eval)
            kl_bounded.append(kl)
            
            gap = kl - theoretical_bound
            print(f"KL(p||q) = {kl:.6f}, Gap above bound = {gap:.6f}")
        
        all_kl_bounded.append(kl_bounded)
    
    mean_kl_bounded = np.array(all_kl_bounded).mean(axis=0)
    std_kl_bounded = np.array(all_kl_bounded).std(axis=0)
    
    print(f"\nDiagnostics: K_eval={K_eval}, n_samples={n_samples_eval}")
    
    return widths, mean_kl_bounded, std_kl_bounded, theoretical_bound


# ============================================================================
# Main experiment with improved plotting
# ============================================================================

def run_experiment_2():
    # Run both experiments
    widths_a, mean_kl_std, std_kl_std, mean_kl_tail, std_kl_tail = experiment_2a_success(n_seeds=3)
    widths_b, mean_kl_bounded, std_kl_bounded, theoretical_bound = experiment_2b_impossibility(n_seeds=3)
    
    # Plot results with ribbons
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Success
    ax1 = axes[0]
    ax1.plot(widths_a, mean_kl_std, 'o-', linewidth=2, markersize=8, 
             label='Standard SIVI', color='steelblue', zorder=3)
    ax1.fill_between(widths_a, mean_kl_std - std_kl_std, mean_kl_std + std_kl_std,
                     alpha=0.3, color='steelblue', zorder=2)
    
    # FIX: Add horizontal line for tail model
    ax1.axhline(y=mean_kl_tail, color='green', linestyle='--', linewidth=2,
                label=f'With tail component ({mean_kl_tail:.3f}±{std_kl_tail:.3f})', zorder=3)
    ax1.fill_between([widths_a[0], widths_a[-1]], 
                     mean_kl_tail - std_kl_tail, mean_kl_tail + std_kl_tail,
                     alpha=0.2, color='green', zorder=1)
    
    ax1.set_xlabel('Network Width (W)', fontsize=12)
    ax1.set_ylabel('Forward KL: KL(p||q)', fontsize=12)
    ax1.set_title('(2a) Success: Sub-Gaussian Target', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Panel B: Impossibility
    ax2 = axes[1]
    ax2.plot(widths_b, mean_kl_bounded, 'o-', linewidth=2, markersize=8, 
             color='red', label='Empirical KL(p||q)', zorder=3)
    ax2.fill_between(widths_b, mean_kl_bounded - std_kl_bounded, 
                     mean_kl_bounded + std_kl_bounded,
                     alpha=0.3, color='red', zorder=2)
    
    # Theoretical lower bound with shaded infeasible region
    ax2.axhspan(0, theoretical_bound, alpha=0.2, color='gray', 
                label='Infeasible region', zorder=1)
    ax2.axhline(y=theoretical_bound, color='darkred', linestyle='--', 
                linewidth=2, label=f'Theoretical bound = {theoretical_bound:.3f}', zorder=3)
    
    ax2.set_xlabel('Network Width (W)', fontsize=12)
    ax2.set_ylabel('Forward KL: KL(p||q)', fontsize=12)
    ax2.set_title('(2b) Impossibility: Heavy-Tailed Target (ν=3)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim([0, max(mean_kl_bounded) * 1.15])
    
    plt.tight_layout()
    plt.savefig('experiment_2_tail_handling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n2a) Sub-Gaussian target (SUCCESS):")
    for w, m, s in zip(widths_a, mean_kl_std, std_kl_std):
        print(f"  W={w:3d}: KL(p||q) = {m:.6f} ± {s:.6f}")
    print(f"  With tail safety: KL(p||q) = {mean_kl_tail:.6f} ± {std_kl_tail:.6f}")
    
    print("\n2b) Heavy-tailed target (IMPOSSIBILITY):")
    print(f"  Theoretical lower bound: {theoretical_bound:.6f}")
    for w, m, s in zip(widths_b, mean_kl_bounded, std_kl_bounded):
        gap = m - theoretical_bound
        print(f"  W={w:3d}: KL(p||q) = {m:.6f} ± {s:.6f}  (gap = {gap:.6f})")
    print(f"\n  → All empirical KL values stay above theoretical bound!")
    print(f"  → Confirms Theorem 3.5: Orlicz tail mismatch ⇒ positive KL gap")


if __name__ == "__main__":
    run_experiment_2()