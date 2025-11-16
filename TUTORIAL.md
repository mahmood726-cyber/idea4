# BART Meta-Regression Tutorial

This tutorial provides step-by-step examples for using BART meta-regression, from basic analyses to advanced workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Example 1: Linear Meta-Regression](#example-1-linear-meta-regression)
3. [Example 2: Non-Linear Dose-Response](#example-2-non-linear-dose-response)
4. [Example 3: Interaction Detection](#example-3-interaction-detection)
5. [Example 4: Real Data Analysis (BCG Vaccine)](#example-4-real-data-analysis-bcg-vaccine)
6. [Example 5: Model Comparison](#example-5-model-comparison)
7. [Complete Workflow](#complete-workflow)

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bart_meta_regression import BARTMetaRegression, compare_with_proper_meta_regression
from simulation_studies import MetaAnalysisSimulator
from visualization import MetaRegressionVisualizer

# Set random seed for reproducibility
np.random.seed(42)
```

## Example 1: Linear Meta-Regression

This example demonstrates BART on data with linear relationships between moderators and effect sizes.

### Step 1: Generate Data

```python
# Create simulator
simulator = MetaAnalysisSimulator(random_state=42)

# Generate linear scenario
# - 50 studies
# - 3 moderators
# - τ² = 0.05 (moderate heterogeneity)
data = simulator.generate_linear_scenario(
    n_studies=50,
    n_features=3,
    tau2=0.05,
    beta_true=np.array([0.5, -0.3, 0.2])  # True coefficients
)

print(f"Generated {len(data['y'])} studies with {data['X'].shape[1]} moderators")
print(f"True heterogeneity (τ²): {data['tau2']:.4f}")
```

### Step 2: Fit BART Model

```python
# Initialize BART meta-regression
bart = BARTMetaRegression(
    n_trees=50,
    n_draws=2000,
    n_tune=1000,
    estimate_tau=True,
    random_state=42
)

# Fit the model
bart.fit(
    X=data['X'],
    y=data['y'],
    se=data['se'],
    feature_names=['Moderator 1', 'Moderator 2', 'Moderator 3'],
    verbose=True
)

print("\nModel fitted successfully!")
```

### Step 3: Inspect Results

```python
# Model summary
summary = bart.summary(include_convergence=True)
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(summary)

# Heterogeneity statistics
het_stats = bart.heterogeneity_stats()
print("\n" + "="*60)
print("HETEROGENEITY STATISTICS")
print("="*60)
print(f"τ² (estimated): {het_stats['tau2']:.4f}")
print(f"τ² 95% CI: [{het_stats['tau2_lower']:.4f}, {het_stats['tau2_upper']:.4f}]")
print(f"I² (BART): {het_stats['I2_bart']:.1f}%")
print(f"I² (classical): {het_stats['I2_classical']:.1f}%")
print(f"Q statistic: {het_stats['Q_residual']:.2f} (p = {het_stats['Q_pvalue']:.4f})")
```

### Step 4: Variable Importance

```python
# Compute variable importance (fast method)
importance_inc = bart.variable_importance(method='inclusion', verbose=True)
print("\n" + "="*60)
print("VARIABLE IMPORTANCE (Inclusion-based)")
print("="*60)
print(importance_inc.to_string(index=False))

# Plot variable importance
fig = bart.plot_variable_importance()
fig.savefig('ex1_importance.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Step 5: Partial Dependence

```python
# Compute and plot partial dependence for each moderator
for feature_idx in range(3):
    fig = bart.plot_partial_dependence(
        feature_idx,
        grid_resolution=50,
        sample_posterior=True  # Include uncertainty
    )
    fig.savefig(f'ex1_pdp_feature{feature_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
```

### Step 6: Model Diagnostics

```python
# Visualize diagnostics
viz = MetaRegressionVisualizer()
fig = viz.diagnostic_panel(bart)
fig.savefig('ex1_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Example 2: Non-Linear Dose-Response

This example shows how BART automatically detects non-linear relationships.

### Step 1: Generate Non-Linear Data

```python
# Generate data with quadratic dose-response
data_nl = simulator.generate_nonlinear_scenario(
    n_studies=60,
    n_features=4,
    nonlinear_type='quadratic'  # Options: 'quadratic', 'cubic', 'sinusoidal', 'threshold'
)

print(f"Generated {len(data_nl['y'])} studies with non-linear dose-response")
```

### Step 2: Fit BART Model

```python
# Fit BART (will automatically capture non-linearity)
bart_nl = BARTMetaRegression(
    n_trees=75,  # More trees for complex relationships
    n_draws=2000,
    n_tune=1000,
    random_state=42
)

bart_nl.fit(
    X=data_nl['X'],
    y=data_nl['y'],
    se=data_nl['se'],
    feature_names=[f'Moderator {i+1}' for i in range(4)],
    verbose=True
)
```

### Step 3: Compare with Linear Model

```python
# Compare BART vs WLS (linear) meta-regression
comparison = compare_with_proper_meta_regression(
    X=data_nl['X'],
    y=data_nl['y'],
    se=data_nl['se'],
    cv_folds=5,
    random_state=42,
    include_gam=True  # Also compare with GAM
)

print("\n" + "="*60)
print("MODEL COMPARISON: BART vs WLS vs GAM")
print("="*60)
print(comparison.to_string(index=False))
```

### Step 4: Visualize Non-Linear Effects

```python
# Plot partial dependence for the non-linear moderator
fig = bart_nl.plot_partial_dependence(
    0,  # First moderator (non-linear)
    grid_resolution=100,  # Higher resolution for smooth curves
    sample_posterior=True
)
fig.suptitle('Non-Linear Dose-Response Relationship', fontsize=14, fontweight='bold')
fig.savefig('ex2_nonlinear_pdp.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Example 3: Interaction Detection

BART automatically detects interactions between moderators without pre-specification.

### Step 1: Generate Data with Interactions

```python
# Generate data with interaction between moderators
data_int = simulator.generate_interaction_scenario(
    n_studies=70,
    n_features=4,
    interaction_strength=0.6  # Moderate to strong interaction
)

print(f"Generated {len(data_int['y'])} studies with moderator interactions")
```

### Step 2: Fit BART Model

```python
# Fit BART
bart_int = BARTMetaRegression(
    n_trees=75,
    n_draws=2000,
    n_tune=1000,
    random_state=42
)

bart_int.fit(
    X=data_int['X'],
    y=data_int['y'],
    se=data_int['se'],
    feature_names=[f'Mod{i+1}' for i in range(4)],
    verbose=True
)
```

### Step 3: Analyze Interaction Strength

```python
# Compute pairwise interaction strengths
print("\n" + "="*60)
print("PAIRWISE INTERACTION STRENGTHS")
print("="*60)

n_features = data_int['X'].shape[1]
interaction_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(i+1, n_features):
        strength = bart_int.interaction_strength(i, j, grid_resolution=20)
        interaction_matrix[i, j] = strength
        interaction_matrix[j, i] = strength
        print(f"Moderator {i+1} × Moderator {j+1}: {strength:.4f}")

# Visualize interaction matrix
plt.figure(figsize=(8, 6))
plt.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
plt.colorbar(label='Interaction Strength')
plt.xticks(range(n_features), [f'Mod{i+1}' for i in range(n_features)])
plt.yticks(range(n_features), [f'Mod{i+1}' for i in range(n_features)])
plt.title('Moderator Interaction Heatmap', fontweight='bold')

for i in range(n_features):
    for j in range(n_features):
        text = plt.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
plt.savefig('ex3_interaction_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Step 4: 2D Partial Dependence

```python
# Visualize 2D partial dependence for interacting features
# (This shows how effect changes across both moderators)

from bart_meta_regression import plot_2d_partial_dependence

fig = plot_2d_partial_dependence(
    bart_int,
    feature_i=0,  # First moderator
    feature_j=1,  # Second moderator
    grid_resolution=30
)
fig.savefig('ex3_2d_pdp.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Example 4: Real Data Analysis (BCG Vaccine)

This example uses real data from the classic BCG vaccine meta-analysis.

### Step 1: Load BCG Data

```python
# Load BCG vaccine data
bcg_data = pd.read_csv('data/bcg_vaccine.csv')

print("BCG Vaccine Meta-Analysis Data")
print(bcg_data.head())
print(f"\nTotal studies: {len(bcg_data)}")

# Extract effect sizes and moderators
y = bcg_data['log_or'].values  # Log odds ratio
se = bcg_data['se_log_or'].values  # Standard error

# Moderators: latitude, year, allocation method
X = bcg_data[['latitude', 'year', 'allocation']].values

feature_names = ['Latitude', 'Publication Year', 'Allocation Method']
study_names = bcg_data['study'].values
```

### Step 2: Fit BART Model

```python
# Fit BART to BCG data
bart_bcg = BARTMetaRegression(
    n_trees=75,
    n_draws=2000,
    n_tune=1000,
    estimate_tau=True,
    random_state=42
)

bart_bcg.fit(
    X=X,
    y=y,
    se=se,
    feature_names=feature_names,
    verbose=True
)

print("\nBART model fitted to BCG vaccine data!")
```

### Step 3: Analyze Results

```python
# Summary statistics
print("\n" + "="*60)
print("BCG VACCINE META-ANALYSIS RESULTS")
print("="*60)
print(bart_bcg.summary())

# Heterogeneity
het_stats = bart_bcg.heterogeneity_stats()
print(f"\nHeterogeneity (I²): {het_stats['I2_bart']:.1f}%")
print(f"Between-study variance (τ²): {het_stats['tau2']:.4f}")

# Variable importance
importance = bart_bcg.variable_importance(method='permutation', n_repeats=5)
print("\nVariable Importance:")
print(importance.to_string(index=False))
```

### Step 4: Visualize Results

```python
# Forest plot
viz = MetaRegressionVisualizer()
fig = viz.forest_plot(
    study_names=study_names,
    effect_sizes=y,
    standard_errors=se,
    predictions=bart_bcg.predictions_mean,
    title='BCG Vaccine Efficacy: BART Meta-Regression'
)
fig.savefig('ex4_bcg_forest.png', dpi=300, bbox_inches='tight')
plt.close()

# Partial dependence: Latitude effect
fig = bart_bcg.plot_partial_dependence(
    0,  # Latitude
    grid_resolution=50,
    sample_posterior=True
)
fig.suptitle('BCG Vaccine Efficacy vs Latitude', fontsize=14, fontweight='bold')
fig.savefig('ex4_bcg_latitude.png', dpi=300, bbox_inches='tight')
plt.close()

# Funnel plot
fig = viz.funnel_plot(y, se, predictions=bart_bcg.predictions_mean)
fig.savefig('ex4_bcg_funnel.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Step 5: Leave-One-Out Cross-Validation

```python
# Run LOO-CV (parallel for speed)
loo_results = bart_bcg.leave_one_out(verbose=True, n_jobs=-1)

print("\nLOO-CV Results:")
print(f"LOO RMSE: {loo_results['rmse_loo']:.4f}")
print(f"Max influence: {loo_results['influence'].max():.4f}")

# Identify influential studies
influential_idx = np.where(loo_results['influence'] > 0.15)[0]
print(f"\nInfluential studies (influence > 0.15):")
for idx in influential_idx:
    print(f"  {study_names[idx]}: influence = {loo_results['influence'][idx]:.4f}")
```

---

## Example 5: Model Comparison

Comparing BART with traditional linear meta-regression and GAM.

### Step 1: Generate Complex Data

```python
# Generate data with mixed characteristics
data_complex = simulator.generate_nonlinear_scenario(
    n_studies=80,
    n_features=5,
    nonlinear_type='cubic'
)
```

### Step 2: Fit Multiple Models

```python
# BART
bart_model = BARTMetaRegression(n_trees=75, random_state=42)
bart_model.fit(data_complex['X'], data_complex['y'], data_complex['se'], verbose=False)

# Run comparison
comparison = compare_with_proper_meta_regression(
    X=data_complex['X'],
    y=data_complex['y'],
    se=data_complex['se'],
    cv_folds=5,
    random_state=42,
    include_gam=True
)

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(comparison.to_string(index=False))
```

### Step 3: Visualize Comparison

```python
# Extract RMSE values for plotting
models = comparison['Model'].values
rmse_means = []
rmse_stds = []

for rmse_str in comparison['RMSE (mean ± std)'].values:
    mean, std = rmse_str.split(' ± ')
    rmse_means.append(float(mean))
    rmse_stds.append(float(std))

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
bars = ax.bar(x, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)

# Color code
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_ylabel('RMSE (Cross-Validation)', fontweight='bold', fontsize=12)
ax.set_title('Model Comparison: BART vs Traditional Methods',
             fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ex5_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Complete Workflow

This section presents a complete analysis workflow from start to finish.

### 1. Data Preparation

```python
# Load your data
# In this example, we'll use simulated data
simulator = MetaAnalysisSimulator(random_state=42)
data = simulator.generate_complex_scenario(
    n_studies=60,
    n_features=5
)

# Extract components
X = data['X']
y = data['y']
se = data['se']
feature_names = [f'Moderator {i+1}' for i in range(5)]
study_names = [f'Study {i+1}' for i in range(len(y))]
```

### 2. Initial Exploration (Fast Settings)

```python
print("="*60)
print("STEP 1: INITIAL EXPLORATION")
print("="*60)

# Fit with reduced settings for quick exploration
bart_initial = BARTMetaRegression(
    n_trees=30,
    n_draws=1000,
    n_tune=500,
    random_state=42
)

bart_initial.fit(X, y, se, feature_names=feature_names, verbose=True)

# Quick diagnostics
print(bart_initial.summary())
importance_initial = bart_initial.variable_importance(method='inclusion')
print("\nInitial variable importance:")
print(importance_initial.to_string(index=False))

# Select top 3 features
top_features = importance_initial.head(3)['feature'].values
top_indices = [list(feature_names).index(f) for f in top_features]
print(f"\nTop 3 moderators: {', '.join(top_features)}")
```

### 3. Refinement with Selected Features

```python
print("\n" + "="*60)
print("STEP 2: REFINEMENT WITH SELECTED FEATURES")
print("="*60)

# Subset to important features
X_selected = X[:, top_indices]
feature_names_selected = [feature_names[i] for i in top_indices]

# Fit with standard settings
bart_refined = BARTMetaRegression(random_state=42)
bart_refined.fit(
    X_selected,
    y,
    se,
    feature_names=feature_names_selected,
    verbose=True
)

# Partial dependence for selected features
for i, feature in enumerate(feature_names_selected):
    pd_result = bart_refined.partial_dependence(i, sample_posterior=False)
    print(f"\nPartial dependence range for {feature}: "
          f"[{pd_result['pd_mean'].min():.3f}, {pd_result['pd_mean'].max():.3f}]")
```

### 4. Final Analysis (Publication Quality)

```python
print("\n" + "="*60)
print("STEP 3: FINAL ANALYSIS (PUBLICATION QUALITY)")
print("="*60)

# Fit with full settings
bart_final = BARTMetaRegression(
    n_trees=75,
    n_draws=2000,
    n_tune=1000,
    estimate_tau=True,
    random_state=42
)

bart_final.fit(
    X_selected,
    y,
    se,
    feature_names=feature_names_selected,
    verbose=True
)

# Comprehensive diagnostics
print("\n--- Permutation Importance ---")
importance_final = bart_final.variable_importance(
    method='permutation',
    n_repeats=10,
    verbose=True
)
print(importance_final.to_string(index=False))

print("\n--- Leave-One-Out Cross-Validation ---")
loo_results = bart_final.leave_one_out(verbose=True, n_jobs=-1)
print(f"LOO RMSE: {loo_results['rmse_loo']:.4f}")

print("\n--- Heterogeneity Statistics ---")
het_stats = bart_final.heterogeneity_stats()
for key, value in het_stats.items():
    print(f"{key}: {value}")
```

### 5. Publication-Quality Outputs

```python
print("\n" + "="*60)
print("STEP 4: GENERATING PUBLICATION OUTPUTS")
print("="*60)

viz = MetaRegressionVisualizer()

# 1. Forest plot
fig = viz.forest_plot(
    study_names=study_names,
    effect_sizes=y,
    standard_errors=se,
    predictions=bart_final.predictions_mean,
    title='BART Meta-Regression: Forest Plot'
)
fig.savefig('final_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Diagnostic panel
fig = viz.diagnostic_panel(bart_final)
fig.savefig('final_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Partial dependence plots with uncertainty
for i, feature in enumerate(feature_names_selected):
    fig = bart_final.plot_partial_dependence(
        i,
        grid_resolution=50,
        sample_posterior=True
    )
    fig.savefig(f'final_pdp_{feature.replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

# 4. Variable importance plot
fig = bart_final.plot_variable_importance()
fig.savefig('final_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll publication outputs saved!")
print("Analysis complete!")
```

---

## Tips for Success

### 1. Start Simple
- Begin with reduced MCMC settings (n_trees=30, n_draws=1000)
- Use `method='inclusion'` for initial variable importance
- Subset to important features before detailed analysis

### 2. Check Convergence
```python
# Always check convergence diagnostics
diag = bart.convergence_diagnostics
print(f"Max R-hat: {diag['mu_rhat_max']:.4f} (should be < 1.1)")
print(f"Min ESS: {diag['mu_ess_min']:.0f} (should be > 400)")
```

### 3. Use Parallelization
```python
# For LOO-CV with many studies
loo_results = bart.leave_one_out(verbose=True, n_jobs=-1)
```

### 4. Visualize Everything
- Always plot partial dependence for top moderators
- Check diagnostic plots for outliers and poor fit
- Use forest plots to communicate results

### 5. Compare with Traditional Methods
```python
# Validate that BART adds value
comparison = compare_with_proper_meta_regression(X, y, se, cv_folds=5)
```

---

## Additional Resources

- **ADVANCED.md**: Performance optimization and computational details
- **README.md**: Quick reference and installation
- **bcg_vaccine_vignette.py**: Complete real data example
- **example_usage.py**: Additional code examples

For questions or issues, please open a GitHub issue.
