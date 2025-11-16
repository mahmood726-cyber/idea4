"""
Test Suite for BART Meta-Regression v2.1.0 (pytest format)

This test file validates all critical fixes made in response to peer review.
Tests ensure that all previously non-functional methods now work correctly.

Run with: pytest test_bart_fixes_pytest.py -v
Or: pytest test_bart_fixes_pytest.py -v -s  (with output)
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from bart_meta_regression import BARTMetaRegression, compare_with_proper_meta_regression
from simulation_studies import MetaAnalysisSimulator
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture(scope="session")
def simulator(random_seed):
    """Create a meta-analysis simulator for test data generation."""
    return MetaAnalysisSimulator(random_state=random_seed)


@pytest.fixture
def linear_data_small(simulator):
    """Small linear dataset (30 studies, 2 features) for quick tests."""
    return simulator.generate_linear_scenario(n_studies=30, n_features=2)


@pytest.fixture
def linear_data_medium(simulator):
    """Medium linear dataset (40 studies, 3 features) with heterogeneity."""
    return simulator.generate_linear_scenario(
        n_studies=40,
        n_features=3,
        tau2=0.06
    )


@pytest.fixture
def linear_data_with_important_feature(simulator):
    """Linear data where first feature is clearly most important."""
    return simulator.generate_linear_scenario(
        n_studies=30,
        n_features=3,
        beta_true=np.array([0.8, 0.1, 0.05])
    )


@pytest.fixture
def nonlinear_data(simulator):
    """Nonlinear data with quadratic relationship."""
    return simulator.generate_nonlinear_scenario(
        n_studies=40,
        n_features=3,
        nonlinear_type='quadratic'
    )


@pytest.fixture
def linear_data_high_tau(simulator):
    """Linear data with moderate heterogeneity."""
    return simulator.generate_linear_scenario(
        n_studies=35,
        n_features=3,
        tau2=0.08
    )


@pytest.fixture
def loo_data_small(simulator):
    """Very small dataset for LOO-CV tests (faster)."""
    return simulator.generate_linear_scenario(n_studies=15, n_features=2)


@pytest.fixture
def bart_model_small(random_seed):
    """Small BART model for quick tests."""
    return BARTMetaRegression(
        n_trees=25,
        n_draws=500,
        n_tune=500,
        random_state=random_seed
    )


@pytest.fixture
def bart_model_medium(random_seed):
    """Medium BART model with tau estimation."""
    return BARTMetaRegression(
        n_trees=30,
        n_draws=500,
        n_tune=500,
        estimate_tau=True,
        random_state=random_seed
    )


@pytest.fixture
def bart_model_loo(random_seed):
    """Smaller BART model optimized for LOO-CV speed."""
    return BARTMetaRegression(
        n_trees=20,
        n_draws=300,
        n_tune=300,
        random_state=random_seed
    )


# ============================================================================
# Test 1: Hierarchical Model with τ² Estimation
# ============================================================================

class TestHierarchicalModel:
    """Test hierarchical BART model with proper τ² estimation."""

    def test_tau_posterior_computed(self, bart_model_medium, linear_data_medium):
        """Test that τ posterior is computed when estimate_tau=True."""
        bart_model_medium.fit(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            verbose=False
        )

        assert bart_model_medium.tau_posterior is not None, \
            "τ posterior should be computed when estimate_tau=True"

    def test_tau2_positive(self, bart_model_medium, linear_data_medium):
        """Test that estimated τ² is positive."""
        bart_model_medium.fit(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            verbose=False
        )

        tau2_estimated = bart_model_medium.get_tau2()
        assert tau2_estimated > 0, "τ² should be positive"

    def test_tau2_estimation_accuracy(self, bart_model_medium, linear_data_medium):
        """Test that τ² is estimated reasonably close to true value."""
        bart_model_medium.fit(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            verbose=False
        )

        tau2_estimated = bart_model_medium.get_tau2()
        tau2_true = linear_data_medium['tau2']
        relative_error = abs(tau2_estimated - tau2_true) / tau2_true

        assert relative_error < 1.0, \
            f"τ² relative error ({relative_error:.2%}) should be < 100%"


# ============================================================================
# Test 2: Predict Method
# ============================================================================

class TestPredictMethod:
    """Test that predict() method returns valid, varying predictions."""

    def test_predict_returns_correct_shape(self, bart_model_small, linear_data_small):
        """Test that predict() returns correct number of predictions."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        X_test = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        predictions = bart_model_small.predict(X_test)

        assert len(predictions) == 3, \
            f"Should return 3 predictions, got {len(predictions)}"

    def test_predict_varies_with_input(self, bart_model_small, linear_data_small):
        """Test that predictions vary for different inputs."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        X_test = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        predictions = bart_model_small.predict(X_test)

        # Check predictions are different (allowing for numerical precision)
        unique_preds = len(np.unique(np.round(predictions, decimals=3)))

        assert unique_preds >= 2, \
            f"Predictions should vary (only {unique_preds}/3 unique values)"

    def test_predict_with_std(self, bart_model_small, linear_data_small):
        """Test that predict() with return_std=True returns valid uncertainties."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        X_test = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        pred_mean, pred_std = bart_model_small.predict(X_test, return_std=True)

        assert len(pred_std) == 3, "Should return 3 std values"
        assert np.all(pred_std > 0), "Standard deviations should be positive"

    def test_predict_with_samples(self, bart_model_small, linear_data_small):
        """Test that predict() with return_samples=True returns posterior samples."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        X_test = np.array([[0.0, 0.0]])
        pred_mean, pred_samples = bart_model_small.predict(
            X_test,
            return_samples=True
        )

        assert pred_samples.shape[0] > 100, \
            "Should return multiple posterior samples"
        assert pred_samples.shape[1] == 1, \
            "Should have samples for 1 test point"


# ============================================================================
# Test 3: Variable Importance
# ============================================================================

class TestVariableImportance:
    """Test variable importance computation methods."""

    @pytest.mark.slow
    def test_permutation_importance_has_std(
        self,
        bart_model_small,
        linear_data_with_important_feature
    ):
        """Test that permutation importance returns std column."""
        bart_model_small.fit(
            linear_data_with_important_feature['X'],
            linear_data_with_important_feature['y'],
            linear_data_with_important_feature['se'],
            verbose=False
        )

        importance = bart_model_small.variable_importance(
            method='permutation',
            n_repeats=3,
            verbose=False
        )

        assert 'std' in importance.columns, \
            "Permutation importance should have 'std' column"

    @pytest.mark.slow
    def test_permutation_importance_positive(
        self,
        bart_model_small,
        linear_data_with_important_feature
    ):
        """Test that top variable has positive importance."""
        bart_model_small.fit(
            linear_data_with_important_feature['X'],
            linear_data_with_important_feature['y'],
            linear_data_with_important_feature['se'],
            verbose=False
        )

        importance = bart_model_small.variable_importance(
            method='permutation',
            n_repeats=3,
            verbose=False
        )

        assert importance.iloc[0]['importance'] > 0, \
            "Top feature should have positive importance"

    def test_inclusion_importance(self, bart_model_small, linear_data_small):
        """Test inclusion-based importance computation."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        importance = bart_model_small.variable_importance(
            method='inclusion',
            verbose=False
        )

        assert 'importance' in importance.columns, \
            "Should have importance column"
        assert len(importance) == linear_data_small['X'].shape[1], \
            "Should have importance for all features"


# ============================================================================
# Test 4: Partial Dependence
# ============================================================================

class TestPartialDependence:
    """Test partial dependence computation."""

    def test_pd_correct_grid_size(self, bart_model_medium, nonlinear_data):
        """Test that partial dependence uses correct grid resolution."""
        bart_model_medium.fit(
            nonlinear_data['X'],
            nonlinear_data['y'],
            nonlinear_data['se'],
            verbose=False
        )

        pd_results = bart_model_medium.partial_dependence(
            0,
            grid_resolution=20,
            sample_posterior=False
        )

        assert len(pd_results['grid_values']) == 20, \
            "Should have 20 grid points"
        assert len(pd_results['pd_mean']) == 20, \
            "Should have 20 PD mean values"

    def test_pd_shows_variation(self, bart_model_medium, nonlinear_data):
        """Test that PD values vary across grid (not constant)."""
        bart_model_medium.fit(
            nonlinear_data['X'],
            nonlinear_data['y'],
            nonlinear_data['se'],
            verbose=False
        )

        pd_results = bart_model_medium.partial_dependence(
            0,
            grid_resolution=20,
            sample_posterior=False
        )

        pd_mean = pd_results['pd_mean']
        unique_pd = len(np.unique(np.round(pd_mean, decimals=3)))

        assert unique_pd >= 15, \
            f"PD values should vary ({unique_pd}/20 unique)"
        assert pd_mean.std() > 0.01, \
            "PD should show meaningful variation"

    def test_pd_with_posterior_samples(self, bart_model_small, linear_data_small):
        """Test that PD with posterior sampling returns confidence intervals."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        pd_results = bart_model_small.partial_dependence(
            0,
            grid_resolution=10,
            sample_posterior=True
        )

        assert 'pd_lower' in pd_results, \
            "Should have lower confidence bound with sample_posterior=True"
        assert 'pd_upper' in pd_results, \
            "Should have upper confidence bound with sample_posterior=True"


# ============================================================================
# Test 5: Heterogeneity Statistics
# ============================================================================

class TestHeterogeneityStats:
    """Test BART-specific heterogeneity statistics."""

    def test_het_stats_all_components(self, bart_model_medium, linear_data_high_tau):
        """Test that all heterogeneity statistics are computed."""
        bart_model_medium.fit(
            linear_data_high_tau['X'],
            linear_data_high_tau['y'],
            linear_data_high_tau['se'],
            verbose=False
        )

        het_stats = bart_model_medium.heterogeneity_stats()

        required_keys = [
            'tau2', 'tau2_lower', 'tau2_upper',
            'I2_bart', 'I2_classical',
            'Q_residual', 'Q_df', 'Q_pvalue'
        ]

        for key in required_keys:
            assert key in het_stats, f"Missing heterogeneity stat: {key}"

    def test_tau2_in_credible_interval(
        self,
        bart_model_medium,
        linear_data_high_tau
    ):
        """Test that τ² point estimate is within credible interval."""
        bart_model_medium.fit(
            linear_data_high_tau['X'],
            linear_data_high_tau['y'],
            linear_data_high_tau['se'],
            verbose=False
        )

        het_stats = bart_model_medium.heterogeneity_stats()

        assert het_stats['tau2_lower'] <= het_stats['tau2'] <= het_stats['tau2_upper'], \
            "τ² should be within its 95% credible interval"

    def test_I2_in_valid_range(self, bart_model_medium, linear_data_high_tau):
        """Test that I² is in [0, 100] range."""
        bart_model_medium.fit(
            linear_data_high_tau['X'],
            linear_data_high_tau['y'],
            linear_data_high_tau['se'],
            verbose=False
        )

        het_stats = bart_model_medium.heterogeneity_stats()

        assert 0 <= het_stats['I2_bart'] <= 100, \
            f"I² BART ({het_stats['I2_bart']}) should be in [0, 100]"
        assert 0 <= het_stats['I2_classical'] <= 100, \
            f"I² classical ({het_stats['I2_classical']}) should be in [0, 100]"


# ============================================================================
# Test 6: Convergence Diagnostics
# ============================================================================

class TestConvergenceDiagnostics:
    """Test convergence diagnostic computation."""

    def test_convergence_diagnostics_computed(
        self,
        bart_model_small,
        linear_data_small
    ):
        """Test that convergence diagnostics are computed after fitting."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        diag = bart_model_small.convergence_diagnostics

        assert diag is not None, \
            "Convergence diagnostics should be computed"
        assert 'mu_rhat_max' in diag, \
            "Should have μ R-hat diagnostic"
        assert 'mu_ess_min' in diag, \
            "Should have μ ESS diagnostic"

    def test_convergence_rhat_acceptable(
        self,
        bart_model_small,
        linear_data_small
    ):
        """Test that R-hat indicates convergence (< 1.1)."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        diag = bart_model_small.convergence_diagnostics

        assert diag['mu_rhat_max'] < 1.1, \
            f"Poor convergence: max R-hat = {diag['mu_rhat_max']:.4f} (should be < 1.1)"

    def test_convergence_status_set(self, bart_model_medium, linear_data_medium):
        """Test that convergence status is set after fitting."""
        bart_model_medium.fit(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            verbose=False
        )

        assert hasattr(bart_model_medium, 'converged'), \
            "Model should have 'converged' attribute"
        assert isinstance(bart_model_medium.converged, bool), \
            "'converged' should be a boolean"


# ============================================================================
# Test 7: Leave-One-Out Cross-Validation
# ============================================================================

class TestLeaveOneOut:
    """Test leave-one-out cross-validation."""

    @pytest.mark.slow
    def test_loo_correct_number_predictions(
        self,
        bart_model_loo,
        loo_data_small
    ):
        """Test that LOO returns predictions for all studies."""
        bart_model_loo.fit(
            loo_data_small['X'],
            loo_data_small['y'],
            loo_data_small['se'],
            verbose=False
        )

        loo_results = bart_model_loo.leave_one_out(verbose=False)

        n_studies = len(loo_data_small['y'])
        assert len(loo_results['predictions']) == n_studies, \
            f"Should have {n_studies} LOO predictions"

    @pytest.mark.slow
    def test_loo_has_influence_measure(self, bart_model_loo, loo_data_small):
        """Test that LOO computes influence measures."""
        bart_model_loo.fit(
            loo_data_small['X'],
            loo_data_small['y'],
            loo_data_small['se'],
            verbose=False
        )

        loo_results = bart_model_loo.leave_one_out(verbose=False)

        assert 'influence' in loo_results, \
            "LOO should compute influence measures"

    @pytest.mark.slow
    def test_loo_influence_sums_to_one(self, bart_model_loo, loo_data_small):
        """Test that influence measures sum to approximately 1."""
        bart_model_loo.fit(
            loo_data_small['X'],
            loo_data_small['y'],
            loo_data_small['se'],
            verbose=False
        )

        loo_results = bart_model_loo.leave_one_out(verbose=False)

        assert np.allclose(loo_results['influence'].sum(), 1.0, atol=0.01), \
            f"Influence should sum to 1, got {loo_results['influence'].sum():.4f}"


# ============================================================================
# Test 8: Comparison Methods
# ============================================================================

class TestComparisonMethods:
    """Test comparison to other meta-regression methods."""

    @pytest.mark.slow
    def test_comparison_includes_bart_and_wls(self, linear_data_medium):
        """Test that comparison function includes BART and WLS."""
        comparison = compare_with_proper_meta_regression(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            cv_folds=3,
            random_state=42,
            include_gam=False
        )

        models = comparison['Model'].values
        assert 'BART Meta-Regression' in models, \
            "Comparison should include BART"
        assert 'WLS Meta-Regression' in models, \
            "Comparison should include WLS"

    @pytest.mark.slow
    def test_comparison_has_rmse(self, linear_data_medium):
        """Test that comparison includes RMSE metrics."""
        comparison = compare_with_proper_meta_regression(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            cv_folds=3,
            random_state=42,
            include_gam=False
        )

        assert 'RMSE (mean ± std)' in comparison.columns, \
            "Comparison should include RMSE column"


# ============================================================================
# Test 9: Ground Truth Evaluation
# ============================================================================

class TestGroundTruthEvaluation:
    """Test evaluation against true parameters in simulations."""

    def test_simulator_returns_true_theta(self, linear_data_small):
        """Test that simulator returns true θ values."""
        assert 'theta_true' in linear_data_small, \
            "Simulator should return theta_true"
        assert len(linear_data_small['theta_true']) == 30, \
            "theta_true should match number of studies"

    def test_rmse_truth_vs_observed(self, bart_model_small, linear_data_small):
        """Test that RMSE vs truth is lower than vs observed data."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        rmse_truth = np.sqrt(mean_squared_error(
            linear_data_small['theta_true'],
            bart_model_small.predictions_mean
        ))
        rmse_observed = np.sqrt(mean_squared_error(
            linear_data_small['y'],
            bart_model_small.predictions_mean
        ))

        assert rmse_truth < rmse_observed, \
            "RMSE vs truth should be lower than vs observed (which includes noise)"

    def test_r2_against_truth(self, bart_model_small, linear_data_small):
        """Test that model explains substantial variance in true θ."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        r2_truth = r2_score(
            linear_data_small['theta_true'],
            bart_model_small.predictions_mean
        )

        assert r2_truth > 0.5, \
            f"Should explain >50% of true variance, got R² = {r2_truth:.3f}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, bart_model_medium, linear_data_medium):
        """Test complete workflow: fit, predict, diagnostics, heterogeneity."""
        # Fit
        bart_model_medium.fit(
            linear_data_medium['X'],
            linear_data_medium['y'],
            linear_data_medium['se'],
            verbose=False
        )

        # Predict
        predictions = bart_model_medium.predict(linear_data_medium['X'])
        assert len(predictions) == len(linear_data_medium['y'])

        # Diagnostics
        diag = bart_model_medium.convergence_diagnostics
        assert diag is not None

        # Heterogeneity
        het_stats = bart_model_medium.heterogeneity_stats()
        assert 'tau2' in het_stats

        # Variable importance (inclusion method, faster)
        importance = bart_model_medium.variable_importance(
            method='inclusion',
            verbose=False
        )
        assert len(importance) == linear_data_medium['X'].shape[1]

    def test_model_summary(self, bart_model_small, linear_data_small):
        """Test that model summary can be generated."""
        bart_model_small.fit(
            linear_data_small['X'],
            linear_data_small['y'],
            linear_data_small['se'],
            verbose=False
        )

        summary = bart_model_small.summary(include_convergence=True)
        assert summary is not None


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Allow running with: python test_bart_fixes_pytest.py
    pytest.main([__file__, "-v", "-s"])
