# BART Meta-Regression: Complete Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Conceptual Background](#conceptual-background)
3. [Installation and Setup](#installation-and-setup)
4. [Basic Workflow](#basic-workflow)
5. [Advanced Features](#advanced-features)
6. [Interpretation Guide](#interpretation-guide)
7. [Publication Checklist](#publication-checklist)
8. [Case Studies](#case-studies)

---

## 1. Introduction

### What is BART Meta-Regression?

BART (Bayesian Additive Regression Trees) meta-regression is a nonparametric alternative to traditional linear meta-regression. It addresses key limitations of linear meta-regression by:

- **Detecting non-linear relationships** without pre-specification
- **Automatically selecting** important moderators
- **Identifying interactions** between moderators without manual specification
- **Preventing overfitting** through Bayesian regularization

### When Should You Use BART Meta-Regression?

**Use BART when:**
- You suspect non-linear dose-response relationships
- You have many potential moderators and need variable selection
- Interactions between moderators are plausible but unknown
- You want to explore complex moderator patterns
- Your meta-analysis has ≥20 studies with good covariate coverage

**Stick with linear meta-regression when:**
- You have < 20 studies
- You have strong a priori hypotheses about linear effects
- You need simple, interpretable regression coefficients
- You're conducting confirmatory (not exploratory) analysis

---

## 2. Conceptual Background

### Traditional Linear Meta-Regression

In standard meta-regression, we model effect sizes as:

```
θᵢ = β₀ + β₁X₁ᵢ + β₂X₂ᵢ + ... + βₚXₚᵢ + uᵢ + eᵢ
```

Where:
- θᵢ = true effect size for study i
- Xⱼᵢ = moderator j for study i
- βⱼ = regression coefficient (assumes linear effect)
- uᵢ ~ N(0, τ²) = between-study heterogeneity
- eᵢ ~ N(0, σᵢ²) = within-study sampling error

**Limitations:**
1. Assumes linear relationships (β₁X₁)
2. Interactions must be pre-specified (β₃X₁X₂)
3. All moderators are included (no automatic selection)
4. Can overfit with many moderators

### BART Meta-Regression

BART models the effect size as a sum of regression trees:

```
θᵢ = Σₘ g(Xᵢ; Tₘ, Mₘ) + uᵢ + eᵢ
```

Where:
- g(Xᵢ; Tₘ, Mₘ) = tree m's prediction for study i
- Tₘ = tree structure (splits)
- Mₘ = terminal node parameters
- Trees are grown using Bayesian priors that regularize complexity

**Advantages:**
1. No assumption about functional form
2. Automatic interaction detection through tree splits
3. Variable selection through tree pruning
4. Regularization prevents overfitting
5. Uncertainty quantification via Bayesian framework

### Key Concepts

**Tree Ensemble**: BART uses multiple weak learners (small trees) that sum to a flexible function approximator.

**Regularization Prior**:
- α (alpha): Controls tree depth probability (higher = shallower trees)
- β (beta): Power parameter in depth penalty
- These prevent overfitting while maintaining flexibility

**Variance Weighting**: Studies are weighted by inverse variance (1/SE²), giving more weight to precise studies.

---

## 3. Installation and Setup

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# PyMC and BART
pip install pymc>=5.0
pip install pymc-bart arviz

# Additional tools
pip install statsmodels joblib tqdm
```

### Step 2: Verify Installation

```python
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb

print(f"PyMC version: {pm.__version__}")
print(f"PyMC-BART available: {pmb is not None}")
```

### Step 3: Import BART Meta-Regression

```python
from bart_meta_regression import BARTMetaRegression
from visualization import MetaRegressionVisualizer
from simulation_studies import MetaAnalysisSimulator
```

---

## 4. Basic Workflow

### Step 1: Prepare Your Data

Your meta-analytic data should include:

```python
import numpy as np
import pandas as pd

# Example data structure
data = pd.DataFrame({
    'study': ['Smith2020', 'Jones2019', ...],
    'effect_size': [0.45, 0.32, ...],      # Effect sizes (log OR, SMD, etc.)
    'se': [0.12, 0.15, ...],                # Standard errors
    'age': [45, 52, ...],                   # Moderator 1
    'dose': [100, 150, ...],                # Moderator 2
    'duration': [8, 12, ...],               # Moderator 3
    # ... additional moderators
})

# Extract components
X = data[['age', 'dose', 'duration']].values
y = data['effect_size'].values
se = data['se'].values
feature_names = ['age', 'dose', 'duration']
```

### Step 2: Fit BART Model

```python
from bart_meta_regression import BARTMetaRegression

# Initialize BART meta-regression
bart = BARTMetaRegression(
    n_trees=50,              # Number of trees (50-100 typical)
    n_draws=2000,            # Posterior samples (2000-5000)
    n_tune=1000,             # Burn-in (1000-2000)
    alpha=0.95,              # Tree depth prior (0.9-0.99)
    beta=2.0,                # Tree depth power (2.0 typical)
    variance_weighting=True, # Use inverse-variance weighting
    random_state=42          # For reproducibility
)

# Fit the model
bart.fit(
    X=X,
    y=y,
    se=se,
    feature_names=feature_names
)
```

### Step 3: Examine Model Summary

```python
# Print summary statistics
summary = bart.summary()
print(summary)

# Output:
# Metric                    Value
# Number of Studies         45
# Number of Features        3
# R²                        0.7234
# RMSE                      0.1456
# Cochran's Q               67.8934
# Q p-value                 0.0012
# I² (%)                    38.45
# τ²                        0.0432
# H²                        1.625
```

### Step 4: Assess Variable Importance

```python
# Calculate variable importance
importance = bart.variable_importance(method='permutation', n_repeats=10)
print(importance)

# Output:
# feature    importance    std
# dose       0.0234       0.0045
# duration   0.0156       0.0032
# age        0.0089       0.0021

# Visualize
fig = bart.plot_variable_importance()
fig.savefig('importance.png', dpi=300)
```

### Step 5: Generate Diagnostic Plots

```python
from visualization import MetaRegressionVisualizer

viz = MetaRegressionVisualizer()

# Comprehensive diagnostic panel
fig = viz.diagnostic_panel(bart, figsize=(16, 12))
fig.savefig('diagnostics.png', dpi=300)
```

### Step 6: Examine Partial Dependence

```python
# Partial dependence for 'dose'
fig = bart.plot_partial_dependence('dose')
fig.savefig('pdp_dose.png', dpi=300)

# Partial dependence for all features
fig = viz.partial_dependence_grid(
    bart,
    feature_indices=['age', 'dose', 'duration']
)
fig.savefig('pdp_all.png', dpi=300)
```

### Step 7: Create Forest Plot

```python
# Forest plot with BART predictions
fig = viz.forest_plot(
    study_names=data['study'].tolist(),
    effect_sizes=y,
    standard_errors=se,
    predictions=bart.predictions,
    figsize=(12, 10)
)
fig.savefig('forest_plot.png', dpi=300)
```

---

## 5. Advanced Features

### 5.1 Interaction Analysis

```python
# Calculate pairwise interaction strengths
print("Interaction Analysis:")
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        strength = bart.interaction_strength(i, j, n_samples=100)
        print(f"{feature_names[i]} × {feature_names[j]}: {strength:.3f}")

# Visualize interaction matrix
fig = viz.interaction_heatmap(
    X=X,
    feature_names=feature_names
)
fig.savefig('interactions.png', dpi=300)
```

### 5.2 Heterogeneity Assessment

```python
# Detailed heterogeneity statistics
het_stats = bart.heterogeneity_stats()

print(f"Heterogeneity Assessment:")
print(f"  I² = {het_stats['I2']:.2f}% (% of variance due to heterogeneity)")
print(f"  τ² = {het_stats['tau2']:.4f} (between-study variance)")
print(f"  H² = {het_stats['H2']:.3f} (relative excess heterogeneity)")
print(f"  Q = {het_stats['Q']:.2f}, p = {het_stats['p_value']:.4f}")

# Interpretation:
# I² = 0-25%: Low heterogeneity
# I² = 25-50%: Moderate heterogeneity
# I² = 50-75%: Substantial heterogeneity
# I² > 75%: Considerable heterogeneity
```

### 5.3 Model Comparison

```python
from bart_meta_regression import compare_with_linear_meta_regression

# Compare BART vs linear meta-regression
comparison = compare_with_linear_meta_regression(
    X=X,
    y=y,
    se=se,
    cv_folds=5,
    random_state=42
)

print(comparison)

# Visualize comparison
fig = viz.comparative_performance_plot(comparison)
fig.savefig('model_comparison.png', dpi=300)
```

### 5.4 Prediction on New Studies

```python
# Predict effect sizes for new studies
X_new = np.array([
    [50, 125, 10],  # New study 1: age=50, dose=125, duration=10
    [60, 200, 12],  # New study 2: age=60, dose=200, duration=12
])

# Point predictions
predictions = bart.predict(X_new)
print(f"Predicted effects: {predictions}")

# With uncertainty
pred_mean, pred_std = bart.predict(X_new, return_std=True)
print(f"Study 1: {pred_mean[0]:.3f} ± {pred_std[0]:.3f}")
print(f"Study 2: {pred_mean[1]:.3f} ± {pred_std[1]:.3f}")

# With quantiles (credible intervals)
pred_quantiles = bart.predict(X_new, quantiles=[0.025, 0.5, 0.975])
print(f"Study 1 95% CI: [{pred_quantiles['q2.5'][0]:.3f}, {pred_quantiles['q97.5'][0]:.3f}]")
```

### 5.5 Residual Diagnostics

```python
# Plot residuals
fig = bart.plot_residuals(figsize=(12, 5))
fig.savefig('residuals.png', dpi=300)

# Check for outliers
standardized_residuals = bart.residuals / bart.se_train
outliers = np.abs(standardized_residuals) > 3
print(f"Potential outliers: {np.where(outliers)[0]}")
```

---

## 6. Interpretation Guide

### 6.1 Variable Importance

**What it means:**
- Higher values = more important moderators
- Near-zero values = moderator has little effect
- Compares relative contribution to predictions

**How to interpret:**
```
Variable Importance:
  dose:     0.0234  ← Strong effect (highest)
  duration: 0.0156  ← Moderate effect
  age:      0.0089  ← Weak effect
  gender:   0.0012  ← Negligible effect
```

**Reporting:**
> "Variable importance analysis identified dose as the most influential moderator (importance = 0.0234), followed by treatment duration (0.0156). Patient age had a weak effect (0.0089), while gender showed negligible importance (0.0012)."

### 6.2 Partial Dependence Plots

**What it shows:**
- How predicted effect size changes with one moderator
- While averaging over other moderators
- Shows non-linear relationships

**How to interpret:**
- Upward slope = positive association
- Downward slope = negative association
- Flat line = no relationship
- Curved line = non-linear relationship

**Example interpretation:**
```
Partial Dependence Plot for 'Dose':
- Effect increases from dose 0-150 mg (positive slope)
- Effect plateaus at dose 150-250 mg (flat)
- Effect decreases at dose > 250 mg (negative slope)
→ Inverted-U dose-response relationship
```

**Reporting:**
> "Partial dependence analysis revealed a non-linear dose-response relationship. Effect sizes increased up to 150 mg (optimal dose), plateaued between 150-250 mg, and decreased beyond 250 mg, suggesting potential adverse effects at high doses."

### 6.3 Heterogeneity Statistics

**I² Interpretation:**
- 0-40%: Low heterogeneity (might not need meta-regression)
- 30-60%: Moderate heterogeneity (meta-regression appropriate)
- 50-90%: Substantial heterogeneity (meta-regression recommended)
- 75-100%: Considerable heterogeneity (check for outliers)

**τ² Interpretation:**
- Scale-dependent measure of between-study variance
- Compare τ² before/after adding moderators
- Reduction in τ² = explained heterogeneity

**Reporting:**
> "Substantial heterogeneity was observed (I² = 67%, τ² = 0.0432, Q = 78.4, p < 0.001). BART meta-regression explained 45% of this heterogeneity (τ² reduced from 0.079 to 0.043), with dose and duration as primary moderators."

### 6.4 Model Fit Statistics

**R² (Coefficient of Determination):**
- 0.00-0.30: Weak explanatory power
- 0.30-0.50: Moderate explanatory power
- 0.50-0.70: Good explanatory power
- 0.70-1.00: Strong explanatory power

**RMSE (Root Mean Squared Error):**
- Lower is better
- Compare with baseline (no moderators)
- Units same as effect size

---

## 7. Publication Checklist

### Reporting Standards

When publishing BART meta-regression results, report:

#### Methods Section
- [ ] Number of studies and total sample size
- [ ] Effect size metric (log OR, SMD, etc.)
- [ ] Moderators examined and their coding
- [ ] BART model specification:
  - Number of trees
  - Number of posterior samples
  - Prior parameters (α, β)
  - Software version (PyMC-BART version)
- [ ] Variance weighting method
- [ ] Model comparison approach (if used)

#### Results Section
- [ ] Heterogeneity statistics (I², τ², Q, p-value)
- [ ] Model fit (R², RMSE)
- [ ] Variable importance ranking
- [ ] Description of non-linear relationships (if found)
- [ ] Interaction effects (if detected)
- [ ] Comparison with linear meta-regression (recommended)

#### Figures
- [ ] Forest plot (observed vs predicted)
- [ ] Funnel plot (publication bias assessment)
- [ ] Variable importance plot
- [ ] Partial dependence plots (key moderators)
- [ ] Diagnostic plots (at minimum: residuals vs fitted, Q-Q plot)

#### Supplementary Materials
- [ ] Full diagnostic panel
- [ ] Interaction heatmap (if relevant)
- [ ] Model comparison results
- [ ] Sensitivity analyses

### Example Results Paragraph

> "We applied Bayesian Additive Regression Trees (BART) meta-regression to examine moderators of treatment efficacy across 48 studies (N = 12,450). The analysis used 50 trees with 2,000 posterior samples following 1,000 burn-in iterations. Substantial heterogeneity was observed (I² = 68%, τ² = 0.056, Q = 146.7, p < 0.001). BART meta-regression explained 52% of this heterogeneity (R² = 0.72, RMSE = 0.14), significantly outperforming linear meta-regression (R² = 0.45, RMSE = 0.24; ΔRMSE = 0.10, 95% CI [0.06, 0.14]).
>
> Variable importance analysis identified treatment dose (importance = 0.034) and baseline severity (importance = 0.028) as primary moderators. Partial dependence analysis revealed a non-linear dose-response relationship with optimal efficacy at 150-200 mg, beyond which effect sizes declined, suggesting diminishing returns or adverse effects. Patient age showed minimal influence (importance = 0.008). An interaction between dose and baseline severity was detected (H-statistic = 0.42), indicating larger dose effects in patients with higher baseline severity."

---

## 8. Case Studies

### Case Study 1: Antidepressant Efficacy

**Research Question:** Do dose, duration, and patient characteristics moderate antidepressant efficacy?

```python
# Load data (hypothetical)
studies = pd.read_csv('antidepressant_studies.csv')

# Moderators
X = studies[['dose_mg', 'duration_weeks', 'age_mean', 'baseline_severity']].values
y = studies['log_or'].values  # Log odds ratio
se = studies['se_log_or'].values

# Fit BART
bart = BARTMetaRegression(n_trees=75, n_draws=3000, random_state=42)
bart.fit(X, y, se, feature_names=['Dose', 'Duration', 'Age', 'Baseline Severity'])

# Variable importance
print(bart.variable_importance())
# Output: Dose and Baseline Severity most important

# Partial dependence: dose
fig = bart.plot_partial_dependence('Dose')
# Finding: Inverted-U relationship (optimal ~100mg)

# Interaction: Dose × Baseline Severity
strength = bart.interaction_strength('Dose', 'Baseline Severity')
# Finding: Strong interaction (H = 0.51)
```

**Conclusion:** Optimal dose ~100mg, with stronger effects in severe depression.

### Case Study 2: Educational Intervention Meta-Analysis

**Research Question:** How do intervention intensity, duration, and student age affect learning outcomes?

```python
# Moderators
X = studies[['intensity_hours', 'duration_weeks', 'age_years', 'class_size']].values
y = studies['cohens_d'].values  # Standardized mean difference
se = studies['se_d'].values

# Fit and analyze
bart = BARTMetaRegression(random_state=42)
bart.fit(X, y, se, feature_names=['Intensity', 'Duration', 'Age', 'Class Size'])

# Check for non-linearity
fig = viz.partial_dependence_grid(bart, feature_indices=[0, 1, 2, 3])
# Finding: Threshold effect at 20 hours/week intensity
```

**Conclusion:** Interventions need ≥20 hours/week for significant effects; duration less important.

---

## Advanced Topics

### Handling Missing Moderators

```python
from sklearn.impute import SimpleImputer

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Use imputed data
bart.fit(X_imputed, y, se)
```

### Sensitivity Analysis

```python
# Test different prior specifications
for alpha in [0.90, 0.95, 0.99]:
    bart_temp = BARTMetaRegression(alpha=alpha, random_state=42)
    bart_temp.fit(X, y, se)
    print(f"Alpha={alpha}: R²={bart_temp.summary()['Value'][2]}")
```

### Categorical Moderators

```python
# One-hot encode categorical variables
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)
country_encoded = encoder.fit_transform(studies[['country']])

# Combine with continuous moderators
X_combined = np.hstack([X_continuous, country_encoded])
bart.fit(X_combined, y, se)
```

---

## Conclusion

BART meta-regression provides a powerful, flexible framework for exploring moderators in meta-analysis. Key advantages include automatic detection of non-linearities and interactions, natural variable selection, and robust uncertainty quantification.

**Remember:**
- BART is exploratory; validate findings with domain expertise
- Compare with linear meta-regression for robustness
- Report comprehensive diagnostics
- Interpret partial dependence plots carefully
- Consider sample size limitations (≥20 studies recommended)

For questions or issues, consult the README or open a GitHub issue.
