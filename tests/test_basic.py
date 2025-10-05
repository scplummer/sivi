"""
Basic smoke tests for SIVI package.

These tests verify that:
- All modules can be imported
- Basic objects can be created
- Core functions execute without crashing
- Outputs have expected shapes and types

Run with: python tests/test_basic.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from sivi.model import SIVIModel
        from sivi.target import GaussianMixture, NealsFunnel, SoftenedQuadratic
        from sivi.trainer import train, evaluate
        from sivi.utils import (
            plot_training_history,
            plot_2d_comparison,
            compute_moments,
            set_random_seed
        )
        print(" All imports successful")
        return True
    except ImportError as e:
        print(f" Import failed: {e}")
        return False


def test_model_creation():
    """Test that SIVI models can be created."""
    print("\nTesting model creation...")
    
    from sivi.model import SIVIModel
    
    try:
        # Basic model
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        assert model.latent_dim == 2
        assert model.mixing_dim == 4
        print(" Basic model created")
        
        # Model with custom parameters
        model2 = SIVIModel(latent_dim=5, mixing_dim=10, hidden_dim=64, n_layers=4)
        assert model2.latent_dim == 5
        print(" Custom model created")
        
        return True
    except Exception as e:
        print(f" Model creation failed: {e}")
        return False


def test_model_sampling():
    """Test that models can sample."""
    print("\nTesting model sampling...")
    
    from sivi.model import SIVIModel
    
    try:
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        
        # Test sampling
        z, epsilon = model.sample(n_samples=10)
        assert z.shape == (10, 2), f"Expected z.shape=(10, 2), got {z.shape}"
        assert epsilon.shape == (10, 4), f"Expected epsilon.shape=(10, 4), got {epsilon.shape}"
        assert not torch.isnan(z).any(), "NaN detected in samples"
        assert not torch.isinf(z).any(), "Inf detected in samples"
        print(" Sampling works, shapes correct, no NaN/Inf")
        
        return True
    except Exception as e:
        print(f" Sampling failed: {e}")
        return False


def test_model_log_prob():
    """Test that log probability computation works."""
    print("\nTesting log probability...")
    
    from sivi.model import SIVIModel
    
    try:
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        
        z, epsilon = model.sample(n_samples=10)
        log_q = model.log_q_z_given_epsilon(z, epsilon)
        
        assert log_q.shape == (10,), f"Expected log_q.shape=(10,), got {log_q.shape}"
        assert not torch.isnan(log_q).any(), "NaN in log probabilities"
        assert not torch.isinf(log_q).any(), "Inf in log probabilities"
        print(" Log probability computation works")
        
        return True
    except Exception as e:
        print(f" Log probability failed: {e}")
        return False


def test_model_elbo():
    """Test that ELBO computation works."""
    print("\nTesting ELBO computation...")
    
    from sivi.model import SIVIModel
    
    try:
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        
        z, epsilon = model.sample(n_samples=10)
        log_p = -0.5 * (z ** 2).sum(dim=-1)  # Dummy target
        
        elbo = model.importance_weighted_elbo(log_p, z, epsilon, K=5)
        
        assert isinstance(elbo.item(), float), "ELBO should be a scalar"
        assert not torch.isnan(elbo), "NaN in ELBO"
        assert not torch.isinf(elbo), "Inf in ELBO"
        print(f" ELBO computation works (value: {elbo.item():.4f})")
        
        return True
    except Exception as e:
        print(f" ELBO computation failed: {e}")
        return False


def test_targets():
    """Test that target distributions can be created and sampled."""
    print("\nTesting target distributions...")
    
    from sivi.target import GaussianMixture, NealsFunnel, SoftenedQuadratic
    
    try:
        # Gaussian Mixture
        gm = GaussianMixture.create_2d_bimodal(separation=4.0)
        z_gm = gm.sample(20)
        log_p_gm = gm.log_prob(z_gm)
        assert z_gm.shape == (20, 2)
        assert log_p_gm.shape == (20,)
        assert not torch.isnan(log_p_gm).any()
        print(" Gaussian Mixture works")
        
        # Neal's Funnel
        funnel = NealsFunnel(dim=5)
        z_funnel = funnel.sample(20)
        log_p_funnel = funnel.log_prob(z_funnel)
        assert z_funnel.shape == (20, 5)
        assert log_p_funnel.shape == (20,)
        assert not torch.isnan(log_p_funnel).any()
        print(" Neal's Funnel works")
        
        # Softened Quadratic
        sq = SoftenedQuadratic(c=1.0, sigma=0.1)
        z_sq = sq.sample(20)
        log_p_sq = sq.log_prob(z_sq)
        assert z_sq.shape == (20, 2)
        assert log_p_sq.shape == (20,)
        assert not torch.isnan(log_p_sq).any()
        print(" Softened Quadratic works")
        
        return True
    except Exception as e:
        print(f" Target creation/sampling failed: {e}")
        return False


def test_training():
    """Test that training loop executes."""
    print("\nTesting training loop...")
    
    from sivi.model import SIVIModel
    from sivi.target import GaussianMixture
    from sivi.trainer import train
    
    try:
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        target = GaussianMixture.create_2d_bimodal(separation=3.0)
        
        # Short training run
        history = train(
            model=model,
            target=target,
            n_iterations=50,
            n_samples=20,
            K=3,
            learning_rate=1e-2,
            print_every=100  # Don't print during test
        )
        
        assert 'elbo' in history
        assert 'iteration' in history
        assert 'time' in history
        assert len(history['elbo']) == 50
        assert len(history['iteration']) == 50
        print(" Training loop executes")
        
        return True
    except Exception as e:
        print(f" Training failed: {e}")
        return False


def test_evaluation():
    """Test that evaluation works."""
    print("\nTesting evaluation...")
    
    from sivi.model import SIVIModel
    from sivi.target import GaussianMixture
    from sivi.trainer import evaluate
    
    try:
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        target = GaussianMixture.create_2d_bimodal(separation=3.0)
        
        results = evaluate(model, target, n_samples=50, K=5)
        
        assert 'elbo' in results
        assert 'log_p_mean' in results
        assert 'log_p_std' in results
        assert isinstance(results['elbo'], float)
        print(" Evaluation works")
        
        return True
    except Exception as e:
        print(f" Evaluation failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    from sivi.model import SIVIModel
    from sivi.target import GaussianMixture
    from sivi.utils import compute_moments, compare_moments, set_random_seed
    
    try:
        # Test random seed
        set_random_seed(42)
        z1 = torch.randn(10, 2)
        set_random_seed(42)
        z2 = torch.randn(10, 2)
        assert torch.allclose(z1, z2), "Random seed not working"
        print(" Random seed works")
        
        # Test moment computation
        samples = torch.randn(100, 2)
        moments = compute_moments(samples)
        assert 'mean' in moments
        assert 'std' in moments
        assert 'cov' in moments
        assert moments['mean'].shape == (2,)
        print(" Moment computation works")
        
        # Test moment comparison
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        target = GaussianMixture.create_2d_bimodal(separation=3.0)
        results = compare_moments(model, target, n_samples=100)
        assert 'target' in results
        assert 'model' in results
        assert 'mean_error' in results
        print(" Moment comparison works")
        
        return True
    except Exception as e:
        print(f" Utility functions failed: {e}")
        return False


def test_gradients():
    """Test that gradients flow properly."""
    print("\nTesting gradient flow...")
    
    from sivi.model import SIVIModel
    
    try:
        model = SIVIModel(latent_dim=2, mixing_dim=4, hidden_dim=32)
        
        # Sample and compute loss
        z, epsilon = model.sample(n_samples=10)
        log_p = -0.5 * (z ** 2).sum(dim=-1)
        elbo = model.importance_weighted_elbo(log_p, z, epsilon, K=5)
        loss = -elbo
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are not NaN
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any(), "NaN in gradients"
                assert not torch.isinf(param.grad).any(), "Inf in gradients"
        
        assert has_grad, "No gradients computed"
        print(" Gradients flow properly")
        
        return True
    except Exception as e:
        print(f" Gradient test failed: {e}")
        return False


def run_all_tests():
    """Run all smoke tests."""
    print("="*70)
    print("SIVI SMOKE TESTS")
    print("="*70)
    
    tests = [
        test_imports,
        test_model_creation,
        test_model_sampling,
        test_model_log_prob,
        test_model_elbo,
        test_targets,
        test_training,
        test_evaluation,
        test_utils,
        test_gradients
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All smoke tests passed! Package is ready to use.")
        return True
    else:
        print(f"\n {total - passed} test(s) failed. Fix these before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)