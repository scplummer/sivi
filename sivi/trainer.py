"""
Training utilities for SIVI.

Provides a simple training loop with:
- ELBO optimization
- Progress tracking
- Early stopping (optional)
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Callable
import time


def train(
    model,
    target,
    n_iterations: int = 5000,
    n_samples: int = 100,
    K: int = 5,
    learning_rate: float = 1e-3,
    print_every: int = 500,
    early_stop_patience: Optional[int] = None,
    early_stop_threshold: float = 1e-4,
    optimizer_class = optim.Adam,
    scheduler_class = None,
    scheduler_kwargs: Optional[Dict] = None
) -> Dict:
    """
    Train SIVI model to approximate target distribution.
    
    Args:
        model: SIVIModel instance
        target: Target distribution with log_prob(z) method
        n_iterations: Number of training iterations
        n_samples: Number of samples per iteration
        K: Number of importance samples for ELBO estimation
        learning_rate: Learning rate for optimizer
        print_every: Print progress every N iterations
        early_stop_patience: Stop if ELBO doesn't improve for N iterations (None = no early stopping)
        early_stop_threshold: Minimum improvement to count as progress
        optimizer_class: PyTorch optimizer class (default: Adam)
        scheduler_class: Optional learning rate scheduler class
        scheduler_kwargs: Kwargs for scheduler initialization
        
    Returns:
        Dictionary with training history:
            - 'elbo': List of ELBO values
            - 'time': Total training time
            - 'converged': Whether early stopping was triggered
    """
    
    # Setup optimizer
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
    # Setup scheduler if requested
    scheduler = None
    if scheduler_class is not None:
        scheduler_kwargs = scheduler_kwargs or {}
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)
    
    # Training history
    history = {
        'elbo': [],
        'iteration': [],
        'time': 0.0,
        'converged': False
    }
    
    # Early stopping state
    best_elbo = float('-inf')
    patience_counter = 0
    
    # Training loop
    start_time = time.time()
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Sample from variational distribution
        z, epsilon = model.sample(n_samples)
        
        # Compute log probability under target
        log_p = target.log_prob(z)
        
        # Compute ELBO using importance weighting
        elbo = model.importance_weighted_elbo(log_p, z, epsilon, K=K)
        
        # Maximize ELBO = minimize negative ELBO
        loss = -elbo
        loss.backward()
        optimizer.step()
        
        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['elbo'].append(elbo.item())
        history['iteration'].append(iteration)
        
        # Print progress
        if (iteration + 1) % print_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iteration {iteration + 1}/{n_iterations} | "
                  f"ELBO: {elbo.item():.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping check
        if early_stop_patience is not None:
            if elbo.item() > best_elbo + early_stop_threshold:
                best_elbo = elbo.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at iteration {iteration + 1}")
                print(f"Best ELBO: {best_elbo:.4f}")
                history['converged'] = True
                break
    
    history['time'] = time.time() - start_time
    
    return history


def evaluate(
    model,
    target,
    n_samples: int = 1000,
    K: int = 10
) -> Dict:
    """
    Evaluate trained SIVI model.
    
    Args:
        model: Trained SIVIModel instance
        target: Target distribution
        n_samples: Number of samples for evaluation
        K: Number of importance samples
        
    Returns:
        Dictionary with evaluation metrics:
            - 'elbo': ELBO estimate
            - 'log_p_mean': Mean log target probability
            - 'log_p_std': Std of log target probability
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from model
        z, epsilon = model.sample(n_samples)
        
        # Compute log probabilities
        log_p = target.log_prob(z)
        
        # Compute ELBO
        elbo = model.importance_weighted_elbo(log_p, z, epsilon, K=K)
        
        results = {
            'elbo': elbo.item(),
            'log_p_mean': log_p.mean().item(),
            'log_p_std': log_p.std().item(),
        }
    
    model.train()
    return results


def train_with_validation(
    model,
    target,
    n_iterations: int = 5000,
    n_samples: int = 100,
    K: int = 5,
    learning_rate: float = 1e-3,
    print_every: int = 500,
    eval_every: int = 1000,
    eval_samples: int = 1000,
    **kwargs
) -> Dict:
    """
    Train with periodic validation.
    
    Args:
        model: SIVIModel instance
        target: Target distribution
        n_iterations: Number of training iterations
        n_samples: Number of samples per training iteration
        K: Number of importance samples for training
        learning_rate: Learning rate
        print_every: Print progress every N iterations
        eval_every: Evaluate every N iterations
        eval_samples: Number of samples for evaluation
        **kwargs: Additional arguments passed to train()
        
    Returns:
        Dictionary with training and validation history
    """
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_elbo': [],
        'eval_elbo': [],
        'eval_iterations': [],
        'iteration': [],
        'time': 0.0
    }
    
    start_time = time.time()
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Training step
        z, epsilon = model.sample(n_samples)
        log_p = target.log_prob(z)
        elbo = model.importance_weighted_elbo(log_p, z, epsilon, K=K)
        
        loss = -elbo
        loss.backward()
        optimizer.step()
        
        # Record training history
        history['train_elbo'].append(elbo.item())
        history['iteration'].append(iteration)
        
        # Print progress
        if (iteration + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {iteration + 1}/{n_iterations} | "
                  f"Train ELBO: {elbo.item():.4f} | "
                  f"Time: {elapsed:.1f}s")
        
        # Validation
        if (iteration + 1) % eval_every == 0:
            eval_results = evaluate(model, target, n_samples=eval_samples, K=K)
            history['eval_elbo'].append(eval_results['elbo'])
            history['eval_iterations'].append(iteration)
            print(f"  Eval ELBO: {eval_results['elbo']:.4f}")
    
    history['time'] = time.time() - start_time
    
    return history


def test_trainer():
    """Test the training function with a simple example."""
    print("Testing trainer...\n")
    
    # Only import when needed
    from model import SIVIModel
    from target import GaussianMixture
    
    # Create simple setup
    target = GaussianMixture.create_2d_bimodal(separation=3.0)
    model = SIVIModel(latent_dim=2, mixing_dim=2, hidden_dim=32)
    
    print("Training for 100 iterations (quick test)...")
    history = train(
        model, 
        target,
        n_iterations=100,
        n_samples=50,
        K=3,
        learning_rate=1e-2,
        print_every=50
    )
    
    print(f"\n  Training completed in {history['time']:.2f}s")
    print(f"  Initial ELBO: {history['elbo'][0]:.4f}")
    print(f"  Final ELBO: {history['elbo'][-1]:.4f}")
    print(f"  Improvement: {history['elbo'][-1] - history['elbo'][0]:.4f}")
    
    # Test evaluation
    print("\nTesting evaluation...")
    eval_results = evaluate(model, target, n_samples=200, K=5)
    print(f"  Evaluation ELBO: {eval_results['elbo']:.4f}")
    print(f"  Mean log p(z): {eval_results['log_p_mean']:.4f}")
    
    # Test early stopping
    print("\nTesting early stopping...")
    model2 = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
    history2 = train(
        model2,
        target,
        n_iterations=1000,
        n_samples=50,
        K=3,
        learning_rate=1e-2,
        print_every=200,
        early_stop_patience=50
    )
    print(f" Early stopping: converged={history2['converged']}")
    
    print("\n All trainer tests passed")


if __name__ == "__main__":
    test_trainer()