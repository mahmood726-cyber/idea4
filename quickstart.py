"""
Quick Start Script for BART Meta-Regression

This script provides a minimal working example that you can run
immediately to see BART meta-regression in action.

Run with: python quickstart.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BART META-REGRESSION: QUICK START")
print("="*70)
print("\nThis demo will:")
print("  1. Generate simulated meta-analytic data")
print("  2. Fit a BART meta-regression model")
print("  3. Display key results")
print("  4. Create publication-quality figures\n")
print("="*70)

# Step 1: Generate simulated data
print("\n[Step 1/4] Generating simulated meta-analytic data...")

from simulation_studies import MetaAnalysisSimulator

simulator = MetaAnalysisSimulator(random_state=42)

# Generate 50 studies with non-linear dose-response
data = simulator.generate_nonlinear_scenario(
    n_studies=50,
    n_features=4,
    tau2=0.05,
    nonlinear_type='quadratic'
)

print(f"✓ Generated {len(data['y'])} studies with 4 moderators")
print(f"  - Mean effect size: {data['y'].mean():.3f}")
print(f"  - Mean standard error: {data['se'].mean():.3f}")
print(f"  - True heterogeneity (τ²): {data['tau2']:.3f}")

# Step 2: Fit BART model
print("\n[Step 2/4] Fitting BART meta-regression model...")
print("  (This may take 1-2 minutes...)")

from bart_meta_regression import BARTMetaRegression

feature_names = ['Dose (mg)', 'Duration (weeks)', 'Age (years)', 'Baseline Severity']

bart = BARTMetaRegression(
    n_trees=50,
    n_draws=1500,      # Reduced for speed
    n_tune=750,        # Reduced for speed
    random_state=42
)

bart.fit(
    X=data['X'],
    y=data['y'],
    se=data['se'],
    feature_names=feature_names
)

print("✓ Model fitting complete!")

# Step 3: Display results
print("\n[Step 3/4] Analysis Results")
print("="*70)

# Model summary
print("\nMODEL SUMMARY:")
print("-"*70)
summary = bart.summary()
print(summary.to_string(index=False))

# Variable importance
print("\n\nVARIABLE IMPORTANCE:")
print("-"*70)
importance = bart.variable_importance(method='permutation', n_repeats=5)
print(importance.to_string(index=False))

# Heterogeneity
print("\n\nHETEROGENEITY STATISTICS:")
print("-"*70)
het_stats = bart.heterogeneity_stats()
print(f"  I² (% variance from heterogeneity):  {het_stats['I2']:.2f}%")
print(f"  τ² (between-study variance):         {het_stats['tau2']:.4f}")
print(f"  Cochran's Q:                         {het_stats['Q']:.2f}")
print(f"  Q p-value:                           {het_stats['p_value']:.4f}")

if het_stats['I2'] < 30:
    het_interpretation = "Low heterogeneity"
elif het_stats['I2'] < 60:
    het_interpretation = "Moderate heterogeneity"
elif het_stats['I2'] < 75:
    het_interpretation = "Substantial heterogeneity"
else:
    het_interpretation = "Considerable heterogeneity"

print(f"\n  Interpretation: {het_interpretation}")

# Step 4: Create figures
print("\n[Step 4/4] Generating publication-quality figures...")

import matplotlib.pyplot as plt
from visualization import MetaRegressionVisualizer

viz = MetaRegressionVisualizer()

# Figure 1: Forest plot
print("  ▸ Creating forest plot...")
fig1 = viz.forest_plot(
    study_names=data['study_names'],
    effect_sizes=data['y'],
    standard_errors=data['se'],
    predictions=bart.predictions,
    figsize=(12, 10)
)
fig1.savefig('quickstart_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("    ✓ Saved: quickstart_forest_plot.png")

# Figure 2: Variable importance
print("  ▸ Creating variable importance plot...")
fig2 = bart.plot_variable_importance(method='permutation', figsize=(10, 6))
fig2.savefig('quickstart_importance.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("    ✓ Saved: quickstart_importance.png")

# Figure 3: Partial dependence (dose effect)
print("  ▸ Creating dose-response curve...")
fig3 = bart.plot_partial_dependence(0, figsize=(10, 7))
fig3.savefig('quickstart_dose_response.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("    ✓ Saved: quickstart_dose_response.png")

# Figure 4: Diagnostic panel
print("  ▸ Creating diagnostic panel...")
fig4 = viz.diagnostic_panel(bart, figsize=(16, 12))
fig4.savefig('quickstart_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("    ✓ Saved: quickstart_diagnostics.png")

# Figure 5: Funnel plot
print("  ▸ Creating funnel plot...")
fig5 = viz.funnel_plot(
    effect_sizes=data['y'],
    standard_errors=data['se'],
    predictions=bart.predictions,
    figsize=(10, 8)
)
fig5.savefig('quickstart_funnel_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig5)
print("    ✓ Saved: quickstart_funnel_plot.png")

# Summary
print("\n" + "="*70)
print("QUICK START COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  • quickstart_forest_plot.png      - Forest plot with BART predictions")
print("  • quickstart_importance.png       - Variable importance ranking")
print("  • quickstart_dose_response.png    - Non-linear dose-response curve")
print("  • quickstart_diagnostics.png      - Comprehensive diagnostic panel")
print("  • quickstart_funnel_plot.png      - Publication bias assessment")

print("\nKey Findings:")
print(f"  • Model explained {float(summary[summary['Metric'] == 'R²']['Value'].values[0]):.1%} of variance")
print(f"  • Most important moderator: {importance.iloc[0]['feature']}")
print(f"  • Heterogeneity: I² = {het_stats['I2']:.1f}% ({het_interpretation})")
print(f"  • Number of studies: {len(data['y'])}")

print("\nNext Steps:")
print("  1. Review the generated figures")
print("  2. Check tutorial.md for detailed interpretation guide")
print("  3. Run example_usage.py for more comprehensive examples")
print("  4. Apply to your own meta-analytic data!")

print("\n" + "="*70)
print("For questions, see README.md or open a GitHub issue")
print("="*70)
