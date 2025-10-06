"""
Semi-Implicit Variational Inference (SIVI) Model

Implementation based on Yin & Zhou (2018).
The variational distribution is defined as:
    q(z) = ∫ q(z|ε) r(ε) dε

where:
    - r(ε) is the mixing distribution (standard Gaussian)
    - q(z|ε) is the conditional distribution parameterized by neural network
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional


class SIVIModel(nn.Module):
    """
    Semi-Implicit Variational Inference model.
    
    Args:
        latent_dim: Dimension of latent variable z
        mixing_dim: Dimension of mixing variable ε
        hidden_dim: Size of hidden layers in the neural network
        n_layers: Number of hidden layers (default: 2)
        mixing_components: Number of components in mixing distribution (default: 1 for standard Gaussian)
        full_covariance: If True, use full covariance matrix for q(z|ε); if False, use diagonal (default: False)
    """
    
    def __init__(
        self, 
        latent_dim: int, 
        mixing_dim: int, 
        hidden_dim: int = 64,
        n_layers: int = 2,
        mixing_components: int = 1,
        full_covariance: bool = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mixing_dim = mixing_dim
        self.hidden_dim = hidden_dim
        self.mixing_components = mixing_components
        self.full_covariance = full_covariance
        
        if mixing_components == 1:
            # Standard Gaussian mixing distribution r(ε) - standard Gaussian
            self.register_buffer(
                'mixing_mean', 
                torch.zeros(mixing_dim)
            )
            self.register_buffer(
                'mixing_std', 
                torch.ones(mixing_dim)
            )
        else:
            # Mixture of Gaussians for mixing distribution
            # Learnable parameters for mixture components
            self.mixing_logits = nn.Parameter(torch.zeros(mixing_components))
            self.mixing_means = nn.Parameter(torch.randn(mixing_components, mixing_dim) * 0.5)
            self.mixing_log_stds = nn.Parameter(torch.zeros(mixing_components, mixing_dim))
        
        # Neural network: ε → (μ(ε), σ(ε)) or (μ(ε), Cholesky(Σ(ε)))
        # Maps mixing variable to parameters of q(z|ε)
        layers = []
        input_dim = mixing_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            ])
            input_dim = hidden_dim
        
        # Output layer size depends on covariance type
        if full_covariance:
            # Output: mean (latent_dim) + Cholesky factors (latent_dim * (latent_dim + 1) / 2)
            n_cholesky = latent_dim * (latent_dim + 1) // 2
            output_dim = latent_dim + n_cholesky
        else:
            # Output: mean and log_std for each dimension
            output_dim = 2 * latent_dim
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.epsilon_to_params = nn.Sequential(*layers)
        
    def sample_epsilon(self, n_samples: int) -> torch.Tensor:
        """
        Sample from mixing distribution r(ε).
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Tensor of shape (n_samples, mixing_dim)
        """
        if self.mixing_components == 1:
            return torch.randn(n_samples, self.mixing_dim)
        else:
            # Sample from mixture of Gaussians
            # Sample component assignments
            weights = torch.softmax(self.mixing_logits, dim=0)
            components = torch.multinomial(weights, n_samples, replacement=True)
            
            # Sample from assigned components
            samples = torch.zeros(n_samples, self.mixing_dim)
            for k in range(self.mixing_components):
                mask = components == k
                n_k = int(mask.sum().item())
                if n_k > 0:
                    mean_k = self.mixing_means[k]
                    std_k = torch.exp(self.mixing_log_stds[k]) + 1e-6
                    samples[mask] = mean_k + std_k * torch.randn(n_k, self.mixing_dim)
            
            return samples
    
    def get_conditional_params(self, epsilon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute parameters of q(z|ε) from mixing variable ε.
        
        Args:
            epsilon: Tensor of shape (n_samples, mixing_dim)
            
        Returns:
            mean: Tensor of shape (n_samples, latent_dim)
            scale_tril or std: If full_covariance, returns lower triangular Cholesky factor (n_samples, latent_dim, latent_dim)
                               Otherwise, returns diagonal std (n_samples, latent_dim)
        """
        params = self.epsilon_to_params(epsilon)
        
        if self.full_covariance:
            # Split into mean and Cholesky parameters
            mean = params[:, :self.latent_dim]
            cholesky_params = params[:, self.latent_dim:]
            
            # Build lower triangular matrix from parameters
            # For 2D: [L11, L21, L22] → [[L11, 0], [L21, L22]]
            n_samples = epsilon.shape[0]
            L = torch.zeros(n_samples, self.latent_dim, self.latent_dim, device=epsilon.device)
            
            # Fill lower triangular part
            tril_indices = torch.tril_indices(self.latent_dim, self.latent_dim, device=epsilon.device)
            L[:, tril_indices[0], tril_indices[1]] = cholesky_params
            
            # Ensure diagonal is positive using softplus
            diag_indices = torch.arange(self.latent_dim, device=epsilon.device)
            L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(L[:, diag_indices, diag_indices]) + 1e-6
            
            return mean, L
        else:
            # Diagonal covariance (original implementation)
            mean, log_std = torch.chunk(params, 2, dim=-1)
            std = torch.exp(log_std) + 1e-6
            return mean, std
    
    def sample_z_given_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Sample z from q(z|ε) using reparameterization trick.
        
        Args:
            epsilon: Tensor of shape (n_samples, mixing_dim)
            
        Returns:
            z: Tensor of shape (n_samples, latent_dim)
        """
        mean, scale = self.get_conditional_params(epsilon)
        noise = torch.randn_like(mean)
        
        if self.full_covariance:
            # z = μ + L * noise, where L is lower triangular Cholesky factor
            # L has shape (n_samples, latent_dim, latent_dim)
            # noise has shape (n_samples, latent_dim) → reshape to (n_samples, latent_dim, 1)
            z = mean + torch.bmm(scale, noise.unsqueeze(-1)).squeeze(-1)
        else:
            # Diagonal: z = μ + σ * noise
            z = mean + scale * noise
            
        return z
    
    def log_q_z_given_epsilon(self, z: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Compute log q(z|ε).
        
        Args:
            z: Tensor of shape (n_samples, latent_dim)
            epsilon: Tensor of shape (n_samples, mixing_dim)
            
        Returns:
            log_prob: Tensor of shape (n_samples,)
        """
        mean, scale = self.get_conditional_params(epsilon)
        
        if self.full_covariance:
            # Use MultivariateNormal with full covariance
            # scale is lower triangular Cholesky factor L
            dist = torch.distributions.MultivariateNormal(mean, scale_tril=scale)
            log_prob = dist.log_prob(z)
        else:
            # Diagonal covariance
            dist = Normal(mean, scale)
            log_prob = dist.log_prob(z).sum(dim=-1)
            
        return log_prob
    
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the variational distribution q(z).
        
        This performs the full hierarchical sampling:
        1. Sample ε ~ r(ε)
        2. Sample z ~ q(z|ε)
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            z: Samples from q(z), shape (n_samples, latent_dim)
            epsilon: Corresponding mixing variables, shape (n_samples, mixing_dim)
        """
        epsilon = self.sample_epsilon(n_samples)
        z = self.sample_z_given_epsilon(epsilon)
        return z, epsilon
    
    def forward(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - same as sample() for compatibility.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            z: Samples from q(z), shape (n_samples, latent_dim)
            epsilon: Corresponding mixing variables, shape (n_samples, mixing_dim)
        """
        return self.sample(n_samples)
    
    def importance_weighted_elbo(
        self, 
        log_p: torch.Tensor,
        z: torch.Tensor,
        epsilon: torch.Tensor,
        K: int = 1
    ) -> torch.Tensor:
        """
        Compute importance-weighted ELBO estimate.
        
        For SIVI, we use importance sampling to estimate the intractable
        marginal q(z). The ELBO is:
            L = E_q(z)[log p(z) - log q(z)]
        
        We estimate log q(z) ≈ log(1/K ∑ q(z|ε_k)) where ε_k ~ r(ε)
        
        Args:
            log_p: Log target density log p(z), shape (n_samples,)
            z: Latent samples, shape (n_samples, latent_dim)
            epsilon: Mixing variables used to generate z, shape (n_samples, mixing_dim)
            K: Number of importance samples for estimating q(z) (includes original epsilon)
            
        Returns:
            elbo: Scalar ELBO estimate
        """
        n_samples = z.shape[0]
        
        # Sample K-1 additional mixing variables for importance weighting
        # We'll include the original epsilon, so we only need K-1 more
        if K > 1:
            epsilon_is = self.sample_epsilon(n_samples * (K - 1)).reshape(n_samples, K - 1, self.mixing_dim)
            # Combine original epsilon with additional samples
            epsilon_all = torch.cat([epsilon.unsqueeze(1), epsilon_is], dim=1)  # (n_samples, K, mixing_dim)
        else:
            epsilon_all = epsilon.unsqueeze(1)  # (n_samples, 1, mixing_dim)
        
        # Compute log q(z|ε_k) for all K importance samples
        # We need to broadcast z to match epsilon_all
        z_expanded = z.unsqueeze(1).expand(-1, K, -1)  # (n_samples, K, latent_dim)
        epsilon_flat = epsilon_all.reshape(n_samples * K, self.mixing_dim)
        z_flat = z_expanded.reshape(n_samples * K, self.latent_dim)
        
        log_q_z_given_eps = self.log_q_z_given_epsilon(z_flat, epsilon_flat)
        log_q_z_given_eps = log_q_z_given_eps.reshape(n_samples, K)
        
        # Estimate log q(z) using log-sum-exp trick for numerical stability
        # log q(z) ≈ log(1/K ∑ q(z|ε_k)) = log(∑ q(z|ε_k)) - log(K)
        log_q_z = torch.logsumexp(log_q_z_given_eps, dim=1) - torch.log(torch.tensor(K, dtype=torch.float32))
        
        # ELBO = E[log p(z) - log q(z)]
        elbo = torch.mean(log_p - log_q_z)
        
        return elbo


def test_model():
    """Quick test to verify the model works."""
    print("Testing SIVIModel...")
    
    # Create model
    model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
    
    # Test sampling
    z, epsilon = model.sample(n_samples=10)
    assert z.shape == (10, 2), f"Expected shape (10, 2), got {z.shape}"
    assert epsilon.shape == (10, 4), f"Expected shape (10, 4), got {epsilon.shape}"
    print(f" Sampling works: z.shape={z.shape}, epsilon.shape={epsilon.shape}")
    
    # Test log probability
    log_q = model.log_q_z_given_epsilon(z, epsilon)
    assert log_q.shape == (10,), f"Expected shape (10,), got {log_q.shape}"
    print(f" Log probability works: log_q.shape={log_q.shape}")
    
    # Test ELBO (with dummy log p)
    log_p = -0.5 * (z ** 2).sum(dim=-1)  # Dummy target: standard Gaussian
    elbo = model.importance_weighted_elbo(log_p, z, epsilon, K=5)
    print(f" ELBO computation works: elbo={elbo.item():.4f}")
    
    print("\nAll tests passed")


if __name__ == "__main__":
    test_model()