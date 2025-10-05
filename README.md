# SIVI: Semi-Implicit Variational Inference

PyTorch implementation of Semi-Implicit Variational Inference (SIVI) based on [Yin & Zhou (2018)](https://arxiv.org/abs/1805.11183).

This implementation is designed for exploring the theoretical properties and practical limitations of SIVI on various target distributions. 

## Features

- **Clean, modular implementation** of SIVI with importance-weighted ELBO
- **Full covariance support** in q(z|ε) for capturing dependencies
- **Multiple test distributions**: Gaussian mixtures, Neal's funnel, warped Gaussians, etc.
- **Comprehensive visualization tools** for analyzing approximation quality
- **Well-documented examples** for reproducing experiments

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd sivi-project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from sivi.model import SIVIModel
from sivi.target import GaussianMixture
from sivi.trainer import train

# Create a bimodal target
target = GaussianMixture.create_2d_bimodal(separation=4.0)

# Initialize SIVI model with full covariance
model = SIVIModel(
    latent_dim=2,
    mixing_dim=16,
    hidden_dim=128,
    n_layers=3,
    full_covariance=True  # Important for capturing dependencies!
)

# Train
history = train(
    model=model,
    target=target,
    n_iterations=5000,
    n_samples=200,
    K=10,
    learning_rate=1e-3
)
```

## Project Structure

```
sivi/
├── sivi/
│   ├── model.py          # SIVI model implementation
│   ├── target.py         # Target distributions
│   ├── trainer.py        # Training and evaluation
│   └── utils.py          # Visualization and analysis tools
│
├── examples/
│   ├── gaussian_mixture.py      # Multimodal distributions
│   ├── neals_funnel.py          # Hierarchical dependencies
│   ├── softened_quadratic.py    # Bounded curvature test
│   └── warped_gaussian.py       # Polynomial/nonlinear warps
│
├── tests/
│   └── test_basic.py     # Smoke tests
│
├── requirements.txt
└── README.md
```

## Examples

### 1. Gaussian Mixture (Multimodality)

Tests SIVI's ability to capture multiple modes:

```bash
python examples/gaussian_mixture.py
```

**Key finding**: SIVI can capture multimodal distributions, but is sensitive to `mixing_dim` and random seed initialization.

### 2. Neal's Funnel (Hierarchical Structure)

Tests handling of variance-dependent relationships:

```bash
python examples/neals_funnel.py
```

**Key finding**: SIVI captures the funnel structure when given sufficient capacity (high `mixing_dim`, deep network).

### 3. Softened Quadratic (Curvature Test)

Tests whether bounded Hessian enables learning nonlinear dependencies:

```bash
python examples/softened_quadratic.py
```

**Key finding**: Full covariance in q(z|ε) is *essential* for learning functional dependencies. Without it, SIVI fails even on simple relationships.

### 4. Warped Gaussian (Functional Forms)

Tests various functional relationships (polynomial, sine, tanh):

```bash
python examples/warped_gaussian.py
```

Change `warp_choice` in the file to test different warps.

## Key Implementation Details

### Full Covariance

By default, SIVI uses diagonal covariance: q(z|ε) = N(μ(ε), diag(σ(ε)²))

For learning dependencies between dimensions, use full covariance:

```python
model = SIVIModel(..., full_covariance=True)
```

This parameterizes q(z|ε) = N(μ(ε), Σ(ε)) where Σ = LL^T (Cholesky decomposition).

**Critical**: Full covariance is necessary for capturing z₁ ↔ z₂ dependencies in functional relationships like z₂ = f(z₁).

### Importance-Weighted ELBO

The ELBO is estimated using importance sampling:

```
ELBO = E_q(z)[log p(z) - log q(z)]
log q(z) ≈ log(1/K ∑_k q(z|ε_k))
```

where ε₁ is the mixing variable that generated z, and ε₂...ε_K are additional samples.

**Implementation note**: We include the original ε in the importance sample (not K additional samples), following the standard estimator.

### Hyperparameters

Critical hyperparameters for success:

- **`mixing_dim`**: Higher = more flexible. Use 16-32 for complex targets.
- **`full_covariance`**: Enable for functional dependencies.
- **`K`**: Number of importance samples. 10-20 for training, higher for evaluation.
- **`n_samples`**: Samples per iteration. 200-1000 depending on target complexity.

## Target Distributions

### GaussianMixture
- Tests: Multimodality
- Variants: Bimodal, trimodal, custom

### NealsFunnel
- Tests: Hierarchical structure, variance dependence
- Structure: z₁ ~ N(0,3), z_i ~ N(0, exp(z₁))

### SoftenedQuadratic
- Tests: Bounded curvature, nonlinear dependencies
- Structure: z₁ ~ N(0,1), z₂ = tanh(c·z₁²) + ε
- `linear_test=True` for debugging

### WarpedGaussian
- Tests: Various functional forms
- Types: Polynomial, sine, tanh, sigmoid, exp
- Customizable coefficients and scale parameters

## Running Tests

```bash
# Run smoke tests
python tests/test_basic.py

# Test individual modules
python sivi/model.py
python sivi/target.py
python sivi/trainer.py
python sivi/utils.py
```

## Research Notes

### When SIVI Works Well
-  Multimodal distributions (with sufficient mixing_dim)
-  Hierarchical structures (Neal's funnel)
-  Bounded nonlinear relationships (with full covariance)

### When SIVI Struggles (What we are looking into currently)
- Very widely separated modes (requires high mixing_dim or lucky initialization)
- Functional dependencies without full covariance
- Periodic relationships (under investigation)

### Key Findings
1. **Full covariance is critical** for learning dependencies
2. **mixing_dim** is more important than network depth/width for multimodality
3. **Initialization sensitivity** can cause mode collapse
4. **Bounded Hessian alone is not sufficient** - architectural constraints matter

## Citation

If you use this code for research, please cite the original SIVI paper:

```bibtex
@inproceedings{yin2018semi,
  title={Semi-implicit variational inference},
  author={Yin, Mingzhang and Zhou, Mingyuan},
  booktitle={International Conference on Machine Learning},
  pages={5660--5669},
  year={2018},
  organization={PMLR}
}
```

## License

[Add This]

## Contact

Sean Plummer 
seanp@uark.edu / snplmmr@gmail.com