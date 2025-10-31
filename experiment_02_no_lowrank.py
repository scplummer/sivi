import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.distributions import Normal, MultivariateNormal, StudentT
from scipy.special import gamma as gamma_func
from scipy.stats import t as student_t

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

print(f"Using device: {device}")

# ============================================================================
# Target Distributions
# ============================================================================

class SubGaussianTarget:
    """Simple sub-Gaussian target (Gaussian for stability)"""
    def __init__(self, dim=2):
        self.dim = dim
        self.dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        print(f"Initialized sub-Gaussian target: N(0, I) in {dim}D")
    
    def sample(self, n_samples):
        return self.dist.sample((n_samples,))
    
    def log_prob(self, x):
        return self.dist.log_prob(x)


class HeavyTailedTarget:
    """Heavy-tailed target: Multivariate Student-t"""
    def __init__(self, dim=2, nu=3.0):
        self.dim = dim
        self.nu = nu
        self.marginal_dist = StudentT(df=nu)
        print(f"Initialized heavy-tailed target: Student-t(ν={nu}) in {dim}D")
    
    def sample(self, n_samples):
        return self.marginal_dist.sample((n_samples, self.dim))
    
    def log_prob(self, x):
        return torch.sum(self.marginal_dist.log_prob(x), dim=-1)


# ============================================================================
# SIVI with numerical fixes
# ============================================================================

class SIVI(nn.Module):
    def __init__(self, latent_dim=2, output_dim=2, hidden_dim=64, max_sigma=2.0, 
                 cap_mu=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_sigma = max_sigma
        self.cap_mu = cap_mu  # For heavy-tailed targets
        
        n_hidden_layers = min(4, max(2, int(1 + np.log2(max(hidden_dim / 16, 1)))))
        
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
        
        self.register_buffer('base_loc', torch.zeros(latent_dim))
        self.register_buffer('base_scale', torch.ones(latent_dim))
        
        print(f"Initialized SIVI: width={hidden_dim}, depth={n_hidden_layers+2}, "
              f"max_sigma={max_sigma}, cap_mu={cap_mu}")
    
    def forward(self, x, K=64):
        """Forward with numerical stability fixes"""
        batch_size = x.shape[0]
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        # FIX 5: Cap mu for heavy-tailed targets
        raw_mu = self.mu_net(z_flat)
        if self.cap_mu:
            mu = 20.0 * torch.tanh(raw_mu)
        else:
            mu = raw_mu
        
        # FIX 1: Single clamp on log_sigma before exp
        log_sigma = self.log_sigma_net(z_flat)
        log_sigma = torch.clamp(log_sigma, min=math.log(1e-4), max=math.log(self.max_sigma))
        sigma = torch.exp(log_sigma)
        
        # FIX 4: Compute inverse once for efficiency
        inv_sigma = torch.exp(-log_sigma)
        
        # Reshape
        mu = mu.reshape(batch_size, K, self.output_dim)
        sigma = sigma.reshape(batch_size, K, self.output_dim)
        inv_sigma = inv_sigma.reshape(batch_size, K, self.output_dim)
        log_sigma = log_sigma.reshape(batch_size, K, self.output_dim)
        
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)
        
        # FIX 4: Numerically stable quadratic form
        quad = ((x_expanded - mu) * inv_sigma) ** 2
        const = math.log(2.0 * math.pi)
        log_prob_xz = -0.5 * torch.sum(quad + 2.0 * log_sigma + const, dim=-1)
        
        log_qx = torch.logsumexp(log_prob_xz, dim=1) - math.log(K)
        
        return log_qx


class SIVIWithTailComponent(nn.Module):
    """SIVI with tail component"""
    def __init__(self, latent_dim=2, output_dim=2, hidden_dim=64, 
                 max_sigma=2.0, tail_sigma=5.0, tail_weight=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_sigma = max_sigma
        self.tail_sigma = tail_sigma
        self.tail_weight = tail_weight
        
        n_hidden_layers = min(4, max(2, int(1 + np.log2(max(hidden_dim / 16, 1)))))
        
        # Core SIVI
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
        
        print(f"Initialized SIVI+Tail: width={hidden_dim}, tail_sigma={tail_sigma}, "
              f"tail_weight={tail_weight}")
    
    def forward(self, x, K=64):
        batch_size = x.shape[0]
        base_dist = Normal(self.base_loc, self.base_scale)
        z = base_dist.sample((batch_size, K))
        
        z_flat = z.reshape(-1, self.latent_dim)
        
        # Apply same fixes as SIVI
        raw_mu = self.mu_net(z_flat)
        mu = raw_mu  # No cap needed for sub-Gaussian target
        
        log_sigma = self.log_sigma_net(z_flat)
        log_sigma = torch.clamp(log_sigma, min=math.log(1e-4), max=math.log(self.max_sigma))
        sigma = torch.exp(log_sigma)
        inv_sigma = torch.exp(-log_sigma)
        
        mu = mu.reshape(batch_size, K, self.output_dim)
        inv_sigma = inv_sigma.reshape(batch_size, K, self.output_dim)
        log_sigma = log_sigma.reshape(batch_size, K, self.output_dim)
        
        x_expanded = x.unsqueeze(1).expand(-1, K, -1)
        
        quad = ((x_expanded - mu) * inv_sigma) ** 2
        const = math.log(2.0 * math.pi)
        log_prob_core = -0.5 * torch.sum(quad + 2.0 * log_sigma + const, dim=-1)
        
        log_qx_core = torch.logsumexp(log_prob_core, dim=1) - math.log(K)
        
        # Tail component
        tail_dist = Normal(self.tail_loc, self.tail_scale)
        log_qx_tail = torch.sum(tail_dist.log_prob(x), dim=-1)
        
        # Mixture
        log_qx = torch.logsumexp(
            torch.stack([
                log_qx_core + math.log(1 - self.tail_weight),
                log_qx_tail + math.log(self.tail_weight)
            ], dim=0),
            dim=0
        )
        
        return log_qx


# ============================================================================
# Training with fixed AMP scaler
# ============================================================================

def train_sivi(model, target, n_epochs=2000, batch_size=256, lr=1e-3, K=64, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    # FIX 3: Correct AMP scaler API
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    
    nan_count = 0
    max_nans = 10
    
    for epoch in range(n_epochs):
        model.train()
        x = target.sample(batch_size).to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                log_qx = model(x, K=K)
                
                if torch.isnan(log_qx).any():
                    nan_count += 1
                    if nan_count > max_nans:
                        print(f"  Too many NaNs ({nan_count}), stopping training")
                        break
                    continue
                
                loss = -log_qx.mean()
            
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > max_nans:
                    print(f"  Too many NaN losses, stopping training")
                    break
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            log_qx = model(x, K=K)
            
            if torch.isnan(log_qx).any():
                nan_count += 1
                if nan_count > max_nans:
                    print(f"  Too many NaNs ({nan_count}), stopping training")
                    break
                continue
            
            loss = -log_qx.mean()
            
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > max_nans:
                    print(f"  Too many NaN losses, stopping training")
                    break
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 500 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    if nan_count > 0:
        print(f"  Warning: Encountered {nan_count} NaN batches during training")
    
    return model


# ============================================================================
# FIX 2: Corrected KL estimation without biasing substitution
# ============================================================================

def estimate_forward_kl(model, target, n_samples=10000, K_eval=2048, batch_size=500):
    """
    FIX 2: Drop NaN samples instead of replacing with -100
    This avoids artificially inflating KL estimates
    """
    model.eval()
    
    with torch.no_grad():
        x = target.sample(n_samples).to(device)
        log_px = target.log_prob(x)
        
        log_qx_list = []
        log_px_list = []
        
        for i in range(0, n_samples, batch_size):
            x_batch = x[i:i+batch_size]
            log_px_batch = log_px[i:i+batch_size]
            
            log_qx_batch = model(x_batch, K=K_eval)
            
            # FIX 2: Drop NaN samples, don't replace with -100
            mask = ~torch.isnan(log_qx_batch)
            
            if mask.any():
                log_qx_list.append(log_qx_batch[mask])
                log_px_list.append(log_px_batch[mask])
            
            if not mask.all():
                n_nan = (~mask).sum().item()
                print(f"    Warning: {n_nan}/{len(mask)} NaN samples in batch {i//batch_size}")
        
        if len(log_qx_list) == 0:
            print("    ERROR: All samples resulted in NaN!")
            return float('nan')
        
        log_qx = torch.cat(log_qx_list)
        log_px = torch.cat(log_px_list)
        
        n_valid = len(log_qx)
        n_dropped = n_samples - n_valid
        
        if n_dropped > 0:
            print(f"    Dropped {n_dropped}/{n_samples} samples ({100*n_dropped/n_samples:.1f}%)")
        
        # Abort if too many samples dropped
        if n_dropped > 0.2 * n_samples:
            print(f"    ERROR: Too many samples dropped (>{20}%), results unreliable")
            return float('nan')
        
        kl = (log_px - log_qx).mean()
    
    return kl.item()


# ============================================================================
# Theoretical bound (same as before)
# ============================================================================

def compute_theoretical_kl_lower_bound(nu=3.0, max_sigma=2.0, dim=2, verbose=True):
    if verbose:
        print(f"\nComputing theoretical lower bound:")
        print(f"  Target: Student-t(nu={nu}), dim={dim}")
        print(f"  SIVI: Bounded variance <= {max_sigma}^2")
        print(f"  Note: Using 1D projection (conservative multivariate bound)")
        print("-" * 60)
    
    C_nu = np.sqrt(nu / np.pi) * gamma_func((nu + 1) / 2) / gamma_func(nu / 2)
    
    def p_tail_1d(t):
        return 2 * (1 - student_t.cdf(t, df=nu))
    
    def q_tail_upper_bound_1d(t):
        return 2 * np.exp(-t**2 / (2 * max_sigma**2))
    
    def kl_bernoulli(p, q):
        if q <= 0 or p <= 0:
            return 0
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
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
        print(f"  (Multivariate tails only strengthen this bound)")
        print("-" * 60)
    
    return best_bound


# ============================================================================
# Experiments
# ============================================================================

def experiment_2a_success(n_seeds=3):
    print("\n" + "="*70)
    print("EXPERIMENT 2A: SUCCESS - Sub-Gaussian Target")
    print("="*70)
    
    target = SubGaussianTarget(dim=2)
    widths = [16, 32, 64, 128, 256]
    
    K_train = 64
    K_eval = 2048
    n_samples_eval = 10000  # FIX 8: Increased samples
    
    all_kl_standard = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        kl_standard = []
        for width in widths:
            print(f"\n[Seed {seed+1}/{n_seeds}] Width = {width}")
            model = SIVI(latent_dim=2, output_dim=2, hidden_dim=width, 
                        max_sigma=3.0, cap_mu=False).to(device)
            
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            model = train_sivi(model, target, n_epochs=2000, lr=5e-4, K=K_train, verbose=False)
            
            kl = estimate_forward_kl(model, target, n_samples=n_samples_eval, K_eval=K_eval)
            kl_standard.append(kl)
            print(f"  KL(p||q) = {kl:.6f}")
        
        all_kl_standard.append(kl_standard)
    
    mean_kl_standard = np.array(all_kl_standard).mean(axis=0)
    std_kl_standard = np.array(all_kl_standard).std(axis=0)
    
    # With tail component
    print("\n[Test 2] With tail safety component (W=64)")
    all_kl_tail = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model_tail = SIVIWithTailComponent(
            latent_dim=2, output_dim=2, hidden_dim=64,
            max_sigma=2.0, tail_sigma=5.0, tail_weight=0.15
        ).to(device)
        
        for m in model_tail.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model_tail = train_sivi(model_tail, target, n_epochs=2000, lr=5e-4, K=K_train, verbose=False)
        kl_tail = estimate_forward_kl(model_tail, target, n_samples=n_samples_eval, K_eval=K_eval)
        all_kl_tail.append(kl_tail)
        print(f"  [Seed {seed+1}] KL = {kl_tail:.6f}")
    
    mean_kl_tail = np.mean(all_kl_tail)
    std_kl_tail = np.std(all_kl_tail)
    
    return widths, mean_kl_standard, std_kl_standard, mean_kl_tail, std_kl_tail


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
    n_samples_eval = 15000  # FIX 8: More samples for heavy tails
    
    all_kl_bounded = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        kl_bounded = []
        for width in widths:
            print(f"\n[Seed {seed+1}/{n_seeds}] Width = {width}")
            
            # FIX 5: Enable mu capping for heavy-tailed targets
            model = SIVI(latent_dim=2, output_dim=2, hidden_dim=width, 
                        max_sigma=max_sigma, cap_mu=True).to(device)
            
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            model = train_sivi(model, target, n_epochs=2000, lr=5e-4, K=K_train, verbose=False)
            
            kl = estimate_forward_kl(model, target, n_samples=n_samples_eval, K_eval=K_eval)
            kl_bounded.append(kl)
            
            gap = kl - theoretical_bound if not np.isnan(kl) else float('nan')
            print(f"  KL(p||q) = {kl:.6f}, Gap above bound = {gap:.6f}")
        
        all_kl_bounded.append(kl_bounded)
    
    mean_kl_bounded = np.nanmean(all_kl_bounded, axis=0)
    std_kl_bounded = np.nanstd(all_kl_bounded, axis=0)
    
    return widths, mean_kl_bounded, std_kl_bounded, theoretical_bound


# ============================================================================
# Main with improved plotting
# ============================================================================

def run_experiment_2():
    widths_a, mean_kl_std, std_kl_std, mean_kl_tail, std_kl_tail = experiment_2a_success(n_seeds=3)
    widths_b, mean_kl_bounded, std_kl_bounded, theoretical_bound = experiment_2b_impossibility(n_seeds=3)
    
    # FIX 6-7: Improved plotting with better labels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Success
    ax1 = axes[0]
    ax1.plot(widths_a, mean_kl_std, 'o-', linewidth=2, markersize=8, 
             color='steelblue', zorder=3, label='Standard SIVI')
    ax1.fill_between(widths_a, mean_kl_std - std_kl_std, mean_kl_std + std_kl_std,
                     alpha=0.3, color='steelblue', zorder=2)
    ax1.axhline(y=mean_kl_tail, color='green', linestyle='--', linewidth=2, zorder=3)
    ax1.fill_between([widths_a[0], widths_a[-1]], 
                     mean_kl_tail - std_kl_tail, mean_kl_tail + std_kl_tail,
                     alpha=0.2, color='green', zorder=1, 
                     label=f'With tail component: {mean_kl_tail:.3f}±{std_kl_tail:.3f}')
    
    ax1.set_xlabel('Network Width (W)', fontsize=12)
    ax1.set_ylabel('Forward KL: KL(p||q)', fontsize=12)
    ax1.set_title('(2a) Success: Sub-Gaussian Target', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Panel B: Impossibility with improved shading
    ax2 = axes[1]
    ax2.plot(widths_b, mean_kl_bounded, 'o-', linewidth=2, markersize=8, 
             color='red', label='Empirical KL(p||q)', zorder=3)
    ax2.fill_between(widths_b, mean_kl_bounded - std_kl_bounded, 
                     mean_kl_bounded + std_kl_bounded,
                     alpha=0.3, color='red', zorder=2)
    
    # FIX 6: Better labeling of infeasible region
    ax2.axhspan(0, theoretical_bound, alpha=0.2, color='gray', zorder=1,
                label='Infeasible region\n(below theoretical bound)')
    ax2.axhline(y=theoretical_bound, color='darkred', linestyle='--', 
                linewidth=2, zorder=3,
                label=f'Theoretical bound = {theoretical_bound:.3f}\n(1D projection, conservative)')
    
    ax2.set_xlabel('Network Width (W)', fontsize=12)
    ax2.set_ylabel('Forward KL: KL(p||q)', fontsize=12)
    ax2.set_title('(2b) Impossibility: Heavy-Tailed Target (ν=3)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_ylim([0, max(mean_kl_bounded[~np.isnan(mean_kl_bounded)]) * 1.2 
                     if not np.all(np.isnan(mean_kl_bounded)) else 0.5])
    
    plt.tight_layout()
    plt.savefig('experiment_2_tail_handling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n2a) Sub-Gaussian target (SUCCESS):")
    for w, m, s in zip(widths_a, mean_kl_std, std_kl_std):
        print(f"  W={w:3d}: KL(p||q) = {m:.6f} ± {s:.6f}")
    print(f"  With tail safety: KL(p||q) = {mean_kl_tail:.6f} ± {std_kl_tail:.6f}")
    
    print("\n2b) Heavy-tailed target (IMPOSSIBILITY):")
    print(f"  Theoretical lower bound: {theoretical_bound:.6f}")
    print(f"  (Note: 1D projection bound; multivariate tails only strengthen it)")
    for w, m, s in zip(widths_b, mean_kl_bounded, std_kl_bounded):
        if not np.isnan(m):
            gap = m - theoretical_bound
            print(f"  W={w:3d}: KL(p||q) = {m:.6f} ± {s:.6f}  (gap = {gap:.6f})")
        else:
            print(f"  W={w:3d}: KL(p||q) = NaN (numerical instability)")
    
    all_above = all(m > theoretical_bound for m in mean_kl_bounded if not np.isnan(m))
    if all_above:
        print(f"\n  ✓ All empirical KL values stay above theoretical bound!")
        print(f"  ✓ Confirms Theorem 3.5: Orlicz tail mismatch ⇒ positive KL gap")
    else:
        print(f"\n  ⚠ Some values below bound (possible numerical issues)")


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    run_experiment_2()
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")