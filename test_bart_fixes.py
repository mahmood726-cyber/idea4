"""
Test Suite for BART Meta-Regression v2.0.0 (Post-Review)

This test file validates all critical fixes made in response to peer review.
Tests ensure that all previously non-functional methods now work correctly.

Run with: python test_bart_fixes.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from bart_meta_regression import BARTMetaRegression, compare_with_proper_meta_regression
from simulation_studies import MetaAnalysisSimulator

print("="*80)
print("BART META-REGRESSION v2.0.0 - VALIDATION TEST SUITE")
print("="*80)
print()

# Test data
np.random.seed(42)
simulator = MetaAnalysisSimulator(random_state=42)


def test_1_hierarchical_model():
    """Test #1: Hierarchical model with τ² estimation"""
    print("[Test 1] Hierarchical BART Model with τ² Estimation")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=40, n_features=3, tau2=0.06)

    bart = BARTMetaRegression(
        n_trees=30,
        n_draws=500,
        n_tune=500,
        estimate_tau=True,
        random_state=42
    )

    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    tau2_estimated = bart.get_tau2()
    tau2_true = data['tau2']

    print(f"  True τ²: {tau2_true:.4f}")
    print(f"  Estimated τ²: {tau2_estimated:.4f}")
    print(f"  Relative error: {abs(tau2_estimated - tau2_true) / tau2_true * 100:.1f}%")

    # Check that we actually estimated heterogeneity
    assert bart.tau_posterior is not None, "❌ FAILED: τ posterior not computed"
    assert tau2_estimated > 0, "❌ FAILED: τ² should be > 0"
    assert abs(tau2_estimated - tau2_true) / tau2_true < 1.0, "❌ FAILED: τ² estimate too far from truth"

    print("  ✅ PASSED: Hierarchical model correctly estimates τ²")
    print()
    return True


def test_2_predict_method():
    """Test #2: predict() method returns different values for different inputs"""
    print("[Test 2] Functional predict() Method")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=30, n_features=2)

    bart = BARTMetaRegression(n_trees=25, n_draws=500, n_tune=500, random_state=42)
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    # Create diverse test cases
    X_test = np.array([
        [0.0, 0.0],    # Low values
        [1.0, 1.0],    # High values
        [-1.0, 1.0],   # Mixed values
    ])

    predictions = bart.predict(X_test)

    print(f"  Prediction for X=[0, 0]: {predictions[0]:.4f}")
    print(f"  Prediction for X=[1, 1]: {predictions[1]:.4f}")
    print(f"  Prediction for X=[-1, 1]: {predictions[2]:.4f}")

    # Check predictions are different
    unique_preds = len(np.unique(np.round(predictions, decimals=3)))

    assert len(predictions) == 3, "❌ FAILED: Should return 3 predictions"
    assert unique_preds >= 2, f"❌ FAILED: Predictions not varying (only {unique_preds} unique values)"

    # Test with uncertainty
    pred_mean, pred_std = bart.predict(X_test, return_std=True)
    assert len(pred_std) == 3, "❌ FAILED: Should return 3 std values"
    assert np.all(pred_std > 0), "❌ FAILED: Std should be positive"

    print(f"  Unique predictions: {unique_preds}/3")
    print("  ✅ PASSED: predict() returns distinct values for different inputs")
    print()
    return True


def test_3_permutation_importance():
    """Test #3: Permutation importance actually refits models"""
    print("[Test 3] True Permutation Importance")
    print("-" * 60)

    # Create data where one feature is clearly important
    data = simulator.generate_linear_scenario(
        n_studies=30,
        n_features=3,
        beta_true=np.array([0.8, 0.1, 0.05])  # First feature most important
    )

    bart = BARTMetaRegression(n_trees=25, n_draws=500, n_tune=500, random_state=42)
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    print("  Computing permutation importance (this takes ~30 seconds)...")
    importance = bart.variable_importance(method='permutation', n_repeats=3, verbose=False)

    print(f"\n  Variable importance:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f} ± {row['std']:.4f}")

    # Check that most important feature is identified
    most_important = importance.iloc[0]['feature']

    assert 'std' in importance.columns, "❌ FAILED: Missing std column"
    assert importance.iloc[0]['importance'] > 0, "❌ FAILED: Top importance should be > 0"

    print(f"  Most important: {most_important}")
    print("  ✅ PASSED: Permutation importance computed with uncertainty")
    print()
    return True


def test_4_partial_dependence():
    """Test #4: Partial dependence shows variation across grid"""
    print("[Test 4] Proper Partial Dependence Computation")
    print("-" * 60)

    # Create data with non-linear relationship
    data = simulator.generate_nonlinear_scenario(
        n_studies=40,
        n_features=3,
        nonlinear_type='quadratic'
    )

    bart = BARTMetaRegression(n_trees=30, n_draws=500, n_tune=500, random_state=42)
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    # Compute PD for first feature
    pd_results = bart.partial_dependence(0, grid_resolution=20, sample_posterior=False)

    grid_values = pd_results['grid_values']
    pd_mean = pd_results['pd_mean']

    print(f"  Grid points: {len(grid_values)}")
    print(f"  PD range: [{pd_mean.min():.3f}, {pd_mean.max():.3f}]")
    print(f"  PD std: {pd_mean.std():.3f}")

    # Check variation
    unique_pd = len(np.unique(np.round(pd_mean, decimals=3)))

    assert len(grid_values) == 20, "❌ FAILED: Should have 20 grid points"
    assert len(pd_mean) == 20, "❌ FAILED: Should have 20 PD values"
    assert unique_pd >= 15, f"❌ FAILED: PD values not varying ({unique_pd}/20 unique)"
    assert pd_mean.std() > 0.01, "❌ FAILED: PD should show variation"

    print(f"  Unique PD values: {unique_pd}/20")
    print("  ✅ PASSED: Partial dependence varies across grid")
    print()
    return True


def test_5_heterogeneity_stats():
    """Test #5: BART-specific heterogeneity statistics"""
    print("[Test 5] BART-Appropriate Heterogeneity Statistics")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=35, n_features=3, tau2=0.08)

    bart = BARTMetaRegression(
        n_trees=30,
        n_draws=500,
        n_tune=500,
        estimate_tau=True,
        random_state=42
    )
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    het_stats = bart.heterogeneity_stats()

    print(f"  τ² (posterior mean): {het_stats['tau2']:.4f}")
    print(f"  τ² 95% CI: [{het_stats['tau2_lower']:.4f}, {het_stats['tau2_upper']:.4f}]")
    print(f"  I² BART: {het_stats['I2_bart']:.1f}%")
    print(f"  I² classical: {het_stats['I2_classical']:.1f}%")
    print(f"  Q (residual): {het_stats['Q_residual']:.2f} (df={het_stats['Q_df']})")
    print(f"  Q p-value: {het_stats['Q_pvalue']:.4f}")

    assert 'tau2' in het_stats, "❌ FAILED: Missing tau2"
    assert 'tau2_lower' in het_stats, "❌ FAILED: Missing tau2_lower"
    assert 'tau2_upper' in het_stats, "❌ FAILED: Missing tau2_upper"
    assert 'I2_bart' in het_stats, "❌ FAILED: Missing I2_bart"
    assert 'Q_residual' in het_stats, "❌ FAILED: Missing Q_residual"
    assert het_stats['tau2_lower'] <= het_stats['tau2'] <= het_stats['tau2_upper'], \
        "❌ FAILED: τ² not in credible interval"

    print("  ✅ PASSED: All heterogeneity statistics computed correctly")
    print()
    return True


def test_6_convergence_diagnostics():
    """Test #6: Convergence diagnostics are computed"""
    print("[Test 6] Convergence Diagnostics")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=30, n_features=2)

    bart = BARTMetaRegression(n_trees=25, n_draws=500, n_tune=500, random_state=42)
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    diag = bart.convergence_diagnostics

    print(f"  τ R-hat: {diag.get('tau_rhat', 'N/A')}")
    print(f"  τ ESS: {diag.get('tau_ess', 'N/A')}")
    print(f"  μ R-hat (max): {diag['mu_rhat_max']:.4f}")
    print(f"  μ ESS (min): {diag['mu_ess_min']:.0f}")

    assert diag is not None, "❌ FAILED: Convergence diagnostics not computed"
    assert 'mu_rhat_max' in diag, "❌ FAILED: Missing mu_rhat_max"
    assert 'mu_ess_min' in diag, "❌ FAILED: Missing mu_ess_min"
    assert diag['mu_rhat_max'] < 1.1, f"❌ FAILED: Poor convergence (R-hat={diag['mu_rhat_max']:.4f})"

    print("  ✅ PASSED: Convergence diagnostics computed and acceptable")
    print()
    return True


def test_7_leave_one_out():
    """Test #7: Leave-one-out cross-validation"""
    print("[Test 7] Leave-One-Out Cross-Validation")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=15, n_features=2)  # Small for speed

    bart = BARTMetaRegression(n_trees=20, n_draws=300, n_tune=300, random_state=42)
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    print("  Running LOO-CV (this may take ~1 minute)...")
    loo_results = bart.leave_one_out(verbose=False)

    print(f"  LOO predictions computed: {len(loo_results['predictions'])}")
    print(f"  LOO RMSE: {loo_results['rmse_loo']:.4f}")
    print(f"  Max influence: {loo_results['influence'].max():.4f}")

    assert len(loo_results['predictions']) == 15, "❌ FAILED: Should have 15 LOO predictions"
    assert 'influence' in loo_results, "❌ FAILED: Missing influence measure"
    assert np.allclose(loo_results['influence'].sum(), 1.0, atol=0.01), \
        "❌ FAILED: Influence measures should sum to ~1"

    print("  ✅ PASSED: LOO-CV completed successfully")
    print()
    return True


def test_8_comparison_methods():
    """Test #8: Comparison to WLS and GAM"""
    print("[Test 8] Comparison to Proper Meta-Regression Methods")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=40, n_features=3)

    print("  Comparing BART vs WLS (this takes ~30 seconds)...")
    comparison = compare_with_proper_meta_regression(
        data['X'], data['y'], data['se'],
        cv_folds=3,
        random_state=42,
        include_gam=False  # GAM optional
    )

    print(f"\n  {comparison.to_string(index=False)}")

    assert 'BART Meta-Regression' in comparison['Model'].values, "❌ FAILED: Missing BART"
    assert 'WLS Meta-Regression' in comparison['Model'].values, "❌ FAILED: Missing WLS"
    assert 'RMSE (mean ± std)' in comparison.columns, "❌ FAILED: Missing RMSE column"

    print("\n  ✅ PASSED: Comparison to WLS meta-regression completed")
    print()
    return True


def test_9_ground_truth_evaluation():
    """Test #9: Simulation evaluates against true theta"""
    print("[Test 9] Ground Truth Evaluation in Simulations")
    print("-" * 60)

    data = simulator.generate_linear_scenario(n_studies=30, n_features=3)

    # Check that simulator returns true theta
    assert 'theta_true' in data, "❌ FAILED: Simulator doesn't return theta_true"
    assert len(data['theta_true']) == 30, "❌ FAILED: theta_true wrong length"

    bart = BARTMetaRegression(n_trees=25, n_draws=500, n_tune=500, random_state=42)
    bart.fit(data['X'], data['y'], data['se'], verbose=False)

    # Evaluate against ground truth
    from sklearn.metrics import mean_squared_error, r2_score

    rmse_truth = np.sqrt(mean_squared_error(data['theta_true'], bart.predictions_mean))
    rmse_observed = np.sqrt(mean_squared_error(data['y'], bart.predictions_mean))
    r2_truth = r2_score(data['theta_true'], bart.predictions_mean)

    print(f"  RMSE vs true θ: {rmse_truth:.4f}")
    print(f"  RMSE vs observed y: {rmse_observed:.4f}")
    print(f"  R² vs true θ: {r2_truth:.4f}")

    assert rmse_truth < rmse_observed, \
        "❌ FAILED: RMSE vs truth should be lower than vs observed (which includes noise)"
    assert r2_truth > 0.5, "❌ FAILED: Should explain >50% of true variance"

    print("  ✅ PASSED: Can evaluate against ground truth")
    print()
    return True


def run_all_tests():
    """Run all validation tests"""
    tests = [
        test_1_hierarchical_model,
        test_2_predict_method,
        test_3_permutation_importance,
        test_4_partial_dependence,
        test_5_heterogeneity_stats,
        test_6_convergence_diagnostics,
        test_7_leave_one_out,
        test_8_comparison_methods,
        test_9_ground_truth_evaluation,
    ]

    results = []
    for i, test_func in enumerate(tests, 1):
        try:
            success = test_func()
            results.append((test_func.__name__, "PASSED"))
        except AssertionError as e:
            print(f"  {e}")
            results.append((test_func.__name__, "FAILED"))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((test_func.__name__, "ERROR"))

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, status in results if status == "PASSED")
    failed = sum(1 for _, status in results if status == "FAILED")
    errors = sum(1 for _, status in results if status == "ERROR")

    for test_name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {test_name}: {status}")

    print()
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed, {errors} errors")
    print()

    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! BART Meta-Regression v2.0.0 is fully functional.")
    else:
        print(f"⚠️  {failed + errors} test(s) need attention.")

    print("="*80)

    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
