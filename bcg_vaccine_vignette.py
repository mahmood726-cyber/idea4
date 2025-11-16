"""
BCG Vaccine Meta-Analysis Vignette

This vignette demonstrates BART meta-regression on real data from the classic
BCG vaccine efficacy meta-analysis (Colditz et al., 1994).

The BCG vaccine is used to prevent tuberculosis. This meta-analysis examines
whether vaccine efficacy varies by:
- Latitude (distance from equator, proxy for mycobacterial exposure)
- Publication year (time trends)
- Allocation method (randomized vs systematic)

Reference:
Colditz, G. A., et al. (1994). Efficacy of BCG vaccine in the prevention of
tuberculosis: Meta-analysis of the published literature. JAMA, 271(9), 698-702.

Run this script with: python bcg_vaccine_vignette.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from bart_meta_regression import BARTMetaRegression, compare_with_proper_meta_regression
from visualization import MetaRegressionVisualizer

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("BCG VACCINE META-ANALYSIS: BART META-REGRESSION VIGNETTE")
print("="*80)
print()


def load_bcg_data():
    """Load and prepare BCG vaccine data."""
    print("[1] LOADING BCG VACCINE DATA")
    print("-" * 60)

    # Load data
    bcg_data = pd.read_csv('data/bcg_vaccine.csv')

    print(f"Loaded {len(bcg_data)} studies from the BCG vaccine meta-analysis")
    print("\nFirst few studies:")
    print(bcg_data.head().to_string(index=False))

    # Extract effect sizes and moderators
    y = bcg_data['log_or'].values  # Log odds ratio (negative = protective)
    se = bcg_data['se_log_or'].values  # Standard error

    # Moderators
    # Note: Encoding allocation as binary (random=1, other=0)
    allocation_binary = (bcg_data['allocation'] == 'random').astype(int).values

    X = np.column_stack([
        bcg_data['latitude'].values,
        bcg_data['year'].values,
        allocation_binary
    ])

    feature_names = ['Latitude', 'Publication Year', 'Random Allocation']
    study_names = bcg_data['study'].values

    print(f"\nEffect sizes (log OR): mean={y.mean():.3f}, range=[{y.min():.3f}, {y.max():.3f}]")
    print(f"Standard errors: mean={se.mean():.3f}, range=[{se.min():.3f}, {se.max():.3f}]")
    print(f"\nModerators:")
    print(f"  - Latitude: range=[{X[:, 0].min():.0f}, {X[:, 0].max():.0f}] degrees")
    print(f"  - Year: range=[{X[:, 1].min():.0f}, {X[:, 1].max():.0f}]")
    print(f"  - Random allocation: {allocation_binary.sum()}/{len(allocation_binary)} studies")

    return X, y, se, feature_names, study_names, bcg_data


def exploratory_analysis(y, se):
    """Perform exploratory analysis of BCG data."""
    print("\n[2] EXPLORATORY ANALYSIS")
    print("-" * 60)

    # Simple fixed-effect meta-analysis
    weights = 1.0 / se**2
    pooled_effect = np.sum(y * weights) / np.sum(weights)
    pooled_se = np.sqrt(1.0 / np.sum(weights))

    print(f"Simple pooled effect (fixed-effect):")
    print(f"  Log OR: {pooled_effect:.3f} ± {pooled_se:.3f}")
    print(f"  OR: {np.exp(pooled_effect):.3f} (protective if < 1.0)")

    # Heterogeneity (Cochran's Q)
    Q = np.sum(weights * (y - pooled_effect)**2)
    df = len(y) - 1
    from scipy import stats
    Q_pvalue = 1 - stats.chi2.cdf(Q, df)

    I2 = max(0, 100 * (Q - df) / Q)

    print(f"\nHeterogeneity:")
    print(f"  Q = {Q:.2f}, df = {df}, p = {Q_pvalue:.4f}")
    print(f"  I² = {I2:.1f}%")

    if I2 > 75:
        print(f"  → High heterogeneity! Meta-regression appropriate.")
    elif I2 > 25:
        print(f"  → Moderate heterogeneity. Investigate sources.")


def fit_bart_model(X, y, se, feature_names):
    """Fit BART meta-regression to BCG data."""
    print("\n[3] FITTING BART META-REGRESSION")
    print("-" * 60)

    # Initialize BART
    bart = BARTMetaRegression(
        n_trees=75,
        n_draws=2000,
        n_tune=1000,
        estimate_tau=True,
        random_state=42
    )

    print("Fitting BART model...")
    bart.fit(
        X=X,
        y=y,
        se=se,
        feature_names=feature_names,
        verbose=True
    )

    print("\nModel fitting complete!")

    # Check convergence
    diag = bart.convergence_diagnostics
    print(f"\nConvergence diagnostics:")
    print(f"  Max R-hat: {diag['mu_rhat_max']:.4f} (should be < 1.1)")
    print(f"  Min ESS: {diag['mu_ess_min']:.0f} (should be > 400)")

    if bart.converged:
        print(f"  ✓ Model converged successfully")
    else:
        print(f"  ⚠ Warning: Model may not have fully converged")

    return bart


def analyze_results(bart, X, y, se):
    """Analyze BART meta-regression results."""
    print("\n[4] ANALYZING RESULTS")
    print("-" * 60)

    # Model summary
    print("\nModel Summary:")
    summary = bart.summary(include_convergence=True)
    print(summary)

    # Heterogeneity statistics
    het_stats = bart.heterogeneity_stats()
    print(f"\nHeterogeneity Statistics:")
    print(f"  τ² (BART): {het_stats['tau2']:.4f}")
    print(f"  τ² 95% CI: [{het_stats['tau2_lower']:.4f}, {het_stats['tau2_upper']:.4f}]")
    print(f"  I² (BART): {het_stats['I2_bart']:.1f}%")
    print(f"  I² (classical): {het_stats['I2_classical']:.1f}%")
    print(f"  Q (residual): {het_stats['Q_residual']:.2f}, p = {het_stats['Q_pvalue']:.4f}")

    # Variable importance
    print(f"\nVariable Importance (Permutation-based):")
    print("  Computing... (this may take a few minutes)")
    importance = bart.variable_importance(method='permutation', n_repeats=5, verbose=False)
    print(importance.to_string(index=False))

    # Interpretation
    top_feature = importance.iloc[0]['feature']
    print(f"\n→ Most important moderator: {top_feature}")


def compare_models(X, y, se):
    """Compare BART with traditional methods."""
    print("\n[5] MODEL COMPARISON: BART vs WLS vs GAM")
    print("-" * 60)

    print("Running cross-validation comparison...")
    print("(This may take a few minutes)")

    comparison = compare_with_proper_meta_regression(
        X=X,
        y=y,
        se=se,
        cv_folds=5,
        random_state=42,
        include_gam=True
    )

    print("\nComparison Results:")
    print(comparison.to_string(index=False))

    # Determine best model
    rmse_values = []
    for rmse_str in comparison['RMSE (mean ± std)'].values:
        mean_val = float(rmse_str.split(' ± ')[0])
        rmse_values.append(mean_val)

    best_model_idx = np.argmin(rmse_values)
    best_model = comparison.iloc[best_model_idx]['Model']
    print(f"\n→ Best cross-validation performance: {best_model}")


def create_visualizations(bart, X, y, se, feature_names, study_names):
    """Create publication-quality visualizations."""
    print("\n[6] CREATING VISUALIZATIONS")
    print("-" * 60)

    viz = MetaRegressionVisualizer()

    # 1. Forest plot
    print("  Creating forest plot...")
    fig = viz.forest_plot(
        study_names=study_names,
        effect_sizes=y,
        standard_errors=se,
        predictions=bart.predictions_mean,
        title='BCG Vaccine Efficacy: BART Meta-Regression'
    )
    fig.savefig('bcg_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: bcg_forest_plot.png")

    # 2. Funnel plot
    print("  Creating funnel plot...")
    fig = viz.funnel_plot(
        effect_sizes=y,
        standard_errors=se,
        predictions=bart.predictions_mean,
        title='BCG Vaccine: Funnel Plot'
    )
    fig.savefig('bcg_funnel_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: bcg_funnel_plot.png")

    # 3. Diagnostic panel
    print("  Creating diagnostic panel...")
    fig = viz.diagnostic_panel(bart)
    fig.suptitle('BCG Vaccine BART Model Diagnostics', fontsize=14, fontweight='bold')
    fig.savefig('bcg_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: bcg_diagnostics.png")

    # 4. Variable importance
    print("  Creating variable importance plot...")
    fig = bart.plot_variable_importance()
    fig.savefig('bcg_variable_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: bcg_variable_importance.png")

    # 5. Partial dependence plots
    for i, feature in enumerate(feature_names):
        print(f"  Creating partial dependence plot for {feature}...")
        fig = bart.plot_partial_dependence(
            i,
            grid_resolution=50,
            sample_posterior=True
        )
        fig.suptitle(f'BCG Vaccine Efficacy vs {feature}',
                     fontsize=14, fontweight='bold')

        filename = f'bcg_pdp_{feature.replace(" ", "_").lower()}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {filename}")


def run_loo_cv(bart, study_names):
    """Run leave-one-out cross-validation."""
    print("\n[7] LEAVE-ONE-OUT CROSS-VALIDATION")
    print("-" * 60)

    print("Running LOO-CV (using parallelization)...")
    loo_results = bart.leave_one_out(verbose=True, n_jobs=-1)

    print(f"\nLOO-CV Results:")
    print(f"  LOO RMSE: {loo_results['rmse_loo']:.4f}")

    # Identify influential studies
    influence_threshold = 3 / len(study_names)
    influential_idx = np.where(loo_results['influence'] > influence_threshold)[0]

    print(f"\nInfluential studies (Cook's D-like influence > {influence_threshold:.4f}):")
    if len(influential_idx) > 0:
        for idx in influential_idx:
            print(f"  - {study_names[idx]}: {loo_results['influence'][idx]:.4f}")
            print(f"    LOO error: {loo_results['errors'][idx]:.4f}")
    else:
        print("  (No highly influential studies detected)")


def interpret_results(bart, feature_names, X):
    """Provide substantive interpretation of results."""
    print("\n[8] SUBSTANTIVE INTERPRETATION")
    print("-" * 60)

    print("BART meta-regression reveals:")
    print()

    # Analyze partial dependence for each moderator
    for i, feature in enumerate(feature_names):
        pd_result = bart.partial_dependence(i, sample_posterior=False)

        pd_range = pd_result['pd_mean'].max() - pd_result['pd_mean'].min()
        feature_range = X[:, i].max() - X[:, i].min()

        print(f"{i+1}. {feature}:")
        print(f"   - Partial dependence range: {pd_range:.3f} log OR units")

        if pd_range > 0.5:
            print(f"   → STRONG effect on vaccine efficacy")
        elif pd_range > 0.2:
            print(f"   → MODERATE effect on vaccine efficacy")
        else:
            print(f"   → WEAK effect on vaccine efficacy")

        # Check for non-linearity
        grid = pd_result['grid_values']
        pd_mean = pd_result['pd_mean']

        # Fit linear model to PD curve
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(grid, pd_mean)
        r2_linear = r_value**2

        if r2_linear < 0.9:
            print(f"   → NON-LINEAR relationship detected (R² linear = {r2_linear:.3f})")
        else:
            print(f"   → Approximately LINEAR relationship (R² linear = {r2_linear:.3f})")

        print()

    # Overall conclusions
    het_stats = bart.heterogeneity_stats()
    I2 = het_stats['I2_bart']

    print("CONCLUSIONS:")
    print()
    print(f"1. Substantial heterogeneity exists (I² = {I2:.1f}%)")

    importance = bart.variable_importance(method='inclusion', verbose=False)
    top_moderator = importance.iloc[0]['feature']
    print(f"2. {top_moderator} is the most important moderator of vaccine efficacy")

    pooled_effect = bart.predictions_mean.mean()
    print(f"3. Overall pooled effect (BART): log OR = {pooled_effect:.3f}")
    print(f"   (OR = {np.exp(pooled_effect):.3f}, protective if < 1.0)")

    if pooled_effect < 0:
        reduction = (1 - np.exp(pooled_effect)) * 100
        print(f"   → BCG vaccine associated with ~{reduction:.0f}% reduction in TB risk")


def run_bcg_analysis():
    """Main function to run complete BCG analysis."""
    print("\nStarting BCG vaccine meta-analysis with BART...")
    print()

    # Load data
    X, y, se, feature_names, study_names, bcg_data = load_bcg_data()

    # Exploratory analysis
    exploratory_analysis(y, se)

    # Fit BART model
    bart = fit_bart_model(X, y, se, feature_names)

    # Analyze results
    analyze_results(bart, X, y, se)

    # Compare models
    compare_models(X, y, se)

    # Create visualizations
    create_visualizations(bart, X, y, se, feature_names, study_names)

    # LOO cross-validation
    run_loo_cv(bart, study_names)

    # Interpret results
    interpret_results(bart, feature_names, X)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Generated outputs:")
    print("  - bcg_forest_plot.png")
    print("  - bcg_funnel_plot.png")
    print("  - bcg_diagnostics.png")
    print("  - bcg_variable_importance.png")
    print("  - bcg_pdp_*.png (partial dependence plots)")
    print()
    print("This vignette demonstrates:")
    print("  ✓ Real data analysis with BART meta-regression")
    print("  ✓ Comparison with traditional methods (WLS, GAM)")
    print("  ✓ Comprehensive diagnostics and validation")
    print("  ✓ Publication-quality visualizations")
    print("  ✓ Substantive interpretation of results")
    print()

    return {
        'bart_model': bart,
        'data': {
            'X': X,
            'y': y,
            'se': se,
            'feature_names': feature_names,
            'study_names': study_names
        }
    }


if __name__ == "__main__":
    results = run_bcg_analysis()
