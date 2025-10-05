"""
Target distributions for SIVI experiments.

Each target provides:
- log_prob(z): Unnormalized log probability
- sample(n): Samples from the distribution (if tractable)
- Additional properties for visualization/analysis
"""

import torch
from torch.distributions import Normal, MultivariateNormal
import numpy as np
from typing import Optional


class GaussianMixture:
    """
    Gaussian mixture model - simple but effective test of multimodality.
    
    p(z) = ∑_k w_k N(z | μ_k, Σ_k)
    
    Args:
        means: List of mean vectors, each shape (dim,)
        covs: List of covariance matrices, each shape (dim, dim)
        weights: Mixture weights (must sum to 1)
    """
    
    def __init__(
        self,
        means: list,
        covs: list,
        weights: Optional[list] = None
    ):
        self.means = [torch.tensor(m, dtype=torch.float32) for m in means]
        self.covs = [torch.tensor(c, dtype=torch.float32) for c in covs]
        self.n_components = len(means)
        self.dim = self.means[0].shape[0]
        
        if weights is None:
            weights = [1.0 / self.n_components] * self.n_components
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        # Create component distributions
        self.components = [
            MultivariateNormal(mean, cov) 
            for mean, cov in zip(self.means, self.covs)
        ]
        
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z) = log ∑_k w_k N(z | μ_k, Σ_k).
        
        Args:
            z: Tensor of shape (n_samples, dim)
            
        Returns:
            log_prob: Tensor of shape (n_samples,)
        """
        # Compute log prob for each component
        log_probs = torch.stack([
            comp.log_prob(z) + torch.log(w)
            for comp, w in zip(self.components, self.weights)
        ], dim=1)  # Shape: (n_samples, n_components)
        
        # Log-sum-exp for numerical stability
        return torch.logsumexp(log_probs, dim=1)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the mixture distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: Tensor of shape (n_samples, dim)
        """
        # Sample component assignments
        component_samples = torch.multinomial(
            self.weights, 
            n_samples, 
            replacement=True
        )
        
        # Sample from assigned components
        samples = torch.zeros(n_samples, self.dim)
        for k in range(self.n_components):
            mask = component_samples == k
            n_k = int(mask.sum().item())
            if n_k > 0:
                samples[mask] = self.components[k].sample(torch.Size([n_k]))
        
        return samples
    
    @staticmethod
    def create_2d_bimodal(separation: float = 4.0) -> 'GaussianMixture':
        """
        Create a simple 2D bimodal Gaussian mixture.
        
        Args:
            separation: Distance between modes
            
        Returns:
            GaussianMixture instance
        """
        means = [
            [-separation/2, 0.0],
            [separation/2, 0.0]
        ]
        covs = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ]
        weights = [0.5, 0.5]
        
        return GaussianMixture(means, covs, weights)
    
    @staticmethod
    def create_2d_trimodal() -> 'GaussianMixture':
        """
        Create a 2D trimodal Gaussian mixture arranged in a triangle.
        
        Returns:
            GaussianMixture instance
        """
        r = 3.0  # Radius of arrangement
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        
        means = [
            [r * np.cos(angle), r * np.sin(angle)]
            for angle in angles
        ]
        covs = [[[0.5, 0.0], [0.0, 0.5]] for _ in range(3)]
        weights = [1/3, 1/3, 1/3]
        
        return GaussianMixture(means, covs, weights)


class NealsFunnel:
    """
    Neal's funnel distribution - classic test for hierarchical models.
    
    The funnel is defined as:
        z_1 ~ N(0, 3)
        z_i ~ N(0, exp(z_1)), for i = 2, ..., dim
    
    This creates a funnel-shaped distribution where the variance of z_2:dim
    depends on z_1. Standard mean-field VI struggles with this.
    
    Args:
        dim: Dimensionality (must be >= 2)
        scale: Scale parameter for z_1 (default: 3.0)
    """
    
    def __init__(self, dim: int = 10, scale: float = 3.0):
        assert dim >= 2, "Neal's funnel requires at least 2 dimensions"
        self.dim = dim
        self.scale = scale
        
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z).
        
        Args:
            z: Tensor of shape (n_samples, dim)
            
        Returns:
            log_prob: Tensor of shape (n_samples,)
        """
        z1 = z[:, 0]
        z_rest = z[:, 1:]
        
        # Log p(z_1)
        log_p_z1 = Normal(0, self.scale).log_prob(z1)
        
        # Log p(z_2:dim | z_1)
        # Each z_i ~ N(0, exp(z_1)), so log p = -0.5 * z_i^2 / exp(2*z_1) - z_1 - 0.5*log(2π)
        log_p_rest = -0.5 * (z_rest ** 2) / torch.exp(2 * z1).unsqueeze(-1) - z1.unsqueeze(-1) - 0.5 * np.log(2 * np.pi)
        log_p_rest = log_p_rest.sum(dim=-1)
        
        return log_p_z1 + log_p_rest
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from Neal's funnel.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: Tensor of shape (n_samples, dim)
        """
        # Sample z_1
        z1 = torch.randn(n_samples) * self.scale
        
        # Sample z_2:dim | z_1
        z_rest = torch.randn(n_samples, self.dim - 1) * torch.exp(z1).unsqueeze(-1)
        
        return torch.cat([z1.unsqueeze(-1), z_rest], dim=1)


class SoftenedQuadratic:
    """
    Softened quadratic distribution for testing curvature bounds.
    
    This distribution tests whether bounded Hessian is key to SIVI success:
        z_1 ~ N(0, 1)
        z_2 = tanh(c * z_1^2) + ε, where ε ~ N(0, σ^2)
    
    The parameter c controls curvature. Unlike hard quadratic (z_2 = z_1^2),
    tanh ensures bounded second derivatives for any finite c.
    
    Args:
        c: Curvature parameter (0 = linear, larger = more curved)
        sigma: Noise standard deviation
        linear_test: If True, use z_2 = c * z_1 + ε (for debugging)
    """
    
    def __init__(self, c: float = 1.0, sigma: float = 0.1, linear_test: bool = False):
        self.c = c
        self.sigma = sigma
        self.dim = 2
        self.linear_test = linear_test
        
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z).
        
        Args:
            z: Tensor of shape (n_samples, 2)
            
        Returns:
            log_prob: Tensor of shape (n_samples,)
        """
        z1 = z[:, 0]
        z2 = z[:, 1]
        
        # p(z_1) = N(0, 1)
        log_p_z1 = Normal(0, 1).log_prob(z1)
        
        # p(z_2 | z_1)
        if self.linear_test:
            mean_z2 = self.c * z1
        else:
            mean_z2 = torch.tanh(self.c * z1 ** 2)
        log_p_z2_given_z1 = Normal(mean_z2, self.sigma).log_prob(z2)
        
        return log_p_z1 + log_p_z2_given_z1
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the softened quadratic distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: Tensor of shape (n_samples, 2)
        """
        # Sample z_1
        z1 = torch.randn(n_samples)
        
        # Sample z_2 | z_1
        if self.linear_test:
            mean_z2 = self.c * z1
        else:
            mean_z2 = torch.tanh(self.c * z1 ** 2)
        z2 = mean_z2 + torch.randn(n_samples) * self.sigma
        
        return torch.stack([z1, z2], dim=1)


class WarpedGaussian:
    """
    Warped Gaussian distribution for testing different functional dependencies.
    
    Structure:
        z_1 ~ N(0, 1)
        z_2 = f(z_1) + ε, where ε ~ N(0, σ^2)
    
    The function f can be:
    - Polynomial: f(x) = a₀ + a₁x + a₂x² + a₃x³ + ...
    - Nonlinear: f(x) = sin, tanh, exp, etc.
    
    Args:
        warp_type: Type of warping ('polynomial', 'sin', 'tanh', 'exp', 'sigmoid')
        coeffs: Coefficients for polynomial warping [a₀, a₁, a₂, ...]
        sigma: Noise standard deviation
        scale: Scale parameter for nonlinear functions
    """
    
    def __init__(
        self, 
        warp_type: str = 'polynomial',
        coeffs: list = [],
        sigma: float = 0.1,
        scale: float = 1.0
    ):
        self.warp_type = warp_type
        self.sigma = sigma
        self.scale = scale
        self.dim = 2
        
        # Default coefficients for polynomial
        if coeffs is None:
            if warp_type == 'polynomial':
                self.coeffs = [0.0, 0.5, 0.3]  # Default: 0.5*z + 0.3*z²
            else:
                self.coeffs = []
        else:
            self.coeffs = coeffs
    
    def _warp_function(self, z1: torch.Tensor) -> torch.Tensor:
        """
        Apply warping function f(z_1).
        
        Args:
            z1: Input tensor
            
        Returns:
            Warped values f(z_1)
        """
        if self.warp_type == 'polynomial':
            # f(x) = sum_i coeffs[i] * x^i
            result = torch.zeros_like(z1)
            for i, coeff in enumerate(self.coeffs):
                result += coeff * (z1 ** i)
            return result
            
        elif self.warp_type == 'sin':
            return torch.sin(self.scale * z1)
            
        elif self.warp_type == 'tanh':
            return torch.tanh(self.scale * z1)
            
        elif self.warp_type == 'exp':
            # Use bounded exp to avoid overflow
            return torch.tanh(torch.exp(self.scale * z1) - 1)
            
        elif self.warp_type == 'sigmoid':
            return torch.sigmoid(self.scale * z1)
            
        else:
            raise ValueError(f"Unknown warp_type: {self.warp_type}")
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z).
        
        Args:
            z: Tensor of shape (n_samples, 2)
            
        Returns:
            log_prob: Tensor of shape (n_samples,)
        """
        z1 = z[:, 0]
        z2 = z[:, 1]
        
        # p(z_1) = N(0, 1)
        log_p_z1 = Normal(0, 1).log_prob(z1)
        
        # p(z_2 | z_1) = N(f(z_1), σ²)
        mean_z2 = self._warp_function(z1)
        log_p_z2_given_z1 = Normal(mean_z2, self.sigma).log_prob(z2)
        
        return log_p_z1 + log_p_z2_given_z1
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the warped Gaussian distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            samples: Tensor of shape (n_samples, 2)
        """
        # Sample z_1
        z1 = torch.randn(n_samples)
        
        # Sample z_2 | z_1
        mean_z2 = self._warp_function(z1)
        z2 = mean_z2 + torch.randn(n_samples) * self.sigma
        
        return torch.stack([z1, z2], dim=1)
    
    @staticmethod
    def create_cubic(sigma: float = 0.2) -> 'WarpedGaussian':
        """
        Create cubic polynomial: z_2 = 0.3*z_1 + 0.2*z_1² + 0.1*z_1³ + ε
        
        Args:
            sigma: Noise level
            
        Returns:
            WarpedGaussian instance
        """
        return WarpedGaussian(
            warp_type='polynomial',
            coeffs=[0.0, 0.3, 0.2, 0.1],
            sigma=sigma
        )
    
    @staticmethod
    def create_sine(scale: float = 2.0, sigma: float = 0.2) -> 'WarpedGaussian':
        """
        Create sine warp: z_2 = sin(scale * z_1) + ε
        
        Args:
            scale: Frequency of sine wave
            sigma: Noise level
            
        Returns:
            WarpedGaussian instance
        """
        return WarpedGaussian(
            warp_type='sin',
            scale=scale,
            sigma=sigma
        )
    
    @staticmethod
    def create_tanh(scale: float = 1.5, sigma: float = 0.2) -> 'WarpedGaussian':
        """
        Create tanh warp: z_2 = tanh(scale * z_1) + ε
        
        Args:
            scale: Steepness of tanh
            sigma: Noise level
            
        Returns:
            WarpedGaussian instance
        """
        return WarpedGaussian(
            warp_type='tanh',
            scale=scale,
            sigma=sigma
        )


def test_targets():
    """Test all target distributions."""
    print("Testing target distributions...\n")
    
    # Test Gaussian Mixture
    print("=== Gaussian Mixture ===")
    gm = GaussianMixture.create_2d_bimodal(separation=4.0)
    z_gm = gm.sample(100)
    log_p_gm = gm.log_prob(z_gm)
    print(f" Bimodal: samples shape={z_gm.shape}, log_prob shape={log_p_gm.shape}")
    print(f"  Sample mean: {z_gm.mean(dim=0).numpy()}")
    print(f"  Log prob range: [{log_p_gm.min():.2f}, {log_p_gm.max():.2f}]")
    
    gm_tri = GaussianMixture.create_2d_trimodal()
    z_tri = gm_tri.sample(100)
    print(f" Trimodal: samples shape={z_tri.shape}")
    
    # Test Neal's Funnel
    print("\n=== Neal's Funnel ===")
    funnel = NealsFunnel(dim=5)
    z_funnel = funnel.sample(100)
    log_p_funnel = funnel.log_prob(z_funnel)
    print(f" Samples shape={z_funnel.shape}, log_prob shape={log_p_funnel.shape}")
    print(f"  z_1 range: [{z_funnel[:, 0].min():.2f}, {z_funnel[:, 0].max():.2f}]")
    print(f"  Log prob range: [{log_p_funnel.min():.2f}, {log_p_funnel.max():.2f}]")
    
    # Test Softened Quadratic
    print("\n=== Softened Quadratic ===")
    sq = SoftenedQuadratic(c=1.0, sigma=0.1)
    z_sq = sq.sample(100)
    log_p_sq = sq.log_prob(z_sq)
    print(f" Samples shape={z_sq.shape}, log_prob shape={log_p_sq.shape}")
    print(f"  Correlation(z_1, z_2): {torch.corrcoef(z_sq.T)[0, 1]:.3f}")
    print(f"  Log prob range: [{log_p_sq.min():.2f}, {log_p_sq.max():.2f}]")
    
    # Test Warped Gaussian
    print("\n=== Warped Gaussian ===")
    wg_poly = WarpedGaussian.create_cubic(sigma=0.2)
    z_poly = wg_poly.sample(100)
    log_p_poly = wg_poly.log_prob(z_poly)
    print(f" Cubic polynomial: samples shape={z_poly.shape}")
    print(f"  Correlation(z_1, z_2): {torch.corrcoef(z_poly.T)[0, 1]:.3f}")
    
    wg_sin = WarpedGaussian.create_sine(scale=2.0, sigma=0.2)
    z_sin = wg_sin.sample(100)
    log_p_sin = wg_sin.log_prob(z_sin)
    print(f" Sine warp: samples shape={z_sin.shape}")
    print(f"  Correlation(z_1, z_2): {torch.corrcoef(z_sin.T)[0, 1]:.3f}")
    
    wg_tanh = WarpedGaussian.create_tanh(scale=1.5, sigma=0.2)
    z_tanh = wg_tanh.sample(100)
    log_p_tanh = wg_tanh.log_prob(z_tanh)
    print(f" Tanh warp: samples shape={z_tanh.shape}")
    print(f"  Correlation(z_1, z_2): {torch.corrcoef(z_tanh.T)[0, 1]:.3f}")
    
    print("\n All target tests passed!")


if __name__ == "__main__":
    test_targets()