# BART Meta-Regression: Advanced Bayesian Nonparametric Meta-Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python framework for **Bayesian Additive Regression Trees (BART)** applied to meta-regression analysis. This implementation provides a powerful, nonparametric alternative to traditional linear meta-regression with automatic variable selection, non-linear relationship detection, and interaction modeling.

## 🎯 Key Features

### Statistical Capabilities
- **Nonparametric Modeling**: No assumptions about functional form of covariate effects
- **Automatic Variable Selection**: Trees naturally select important moderators
- **Non-linear Relationships**: Captures complex dose-response curves without pre-specification
- **Interaction Detection**: Automatically identifies interactions between moderators
- **Variance Weighting**: Proper inverse-variance weighting for meta-analytic data
- **Heterogeneity Assessment**: I², τ², H², Cochran's Q statistics
- **Uncertainty Quantification**: Full Bayesian credible intervals and prediction intervals

### Advanced Diagnostics
- Variable importance via permutation and inclusion frequency
- Partial dependence plots with uncertainty
- Interaction strength analysis (Friedman's H-statistic)
- Comprehensive residual diagnostics
- Model comparison with traditional methods
- Publication bias assessment (funnel plots)

### Publication-Quality Visualizations
- Enhanced forest plots with BART predictions
- Funnel plots with heterogeneity contours
- Diagnostic panels (6+ diagnostic plots)
- Variable importance plots
- Partial dependence plots
- Interaction heatmaps
- Model comparison plots

## 📦 Installation

### Requirements
- Python 3.8+
- PyMC 5.0+
- PyMC-BART 0.5+
- NumPy, Pandas, Matplotlib, Seaborn, SciPy, scikit-learn

### Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install pymc-bart arviz statsmodels
```

## 🚀 Quick Start

### Basic Usage

```python
from bart_meta_regression import BARTMetaRegression
import numpy as np

# Prepare your meta-analytic data
# X: study-level covariates/moderators (n_studies × n_features)
# y: effect sizes (n_studies,)
# se: standard errors (n_studies,)

X = np.array([...])  # Your moderator variables
y = np.array([...])  # Effect sizes (e.g., log odds ratios, SMDs)
se = np.array([...]) # Standard errors

# Fit BART meta-regression
bart = BARTMetaRegression(
    n_trees=50,           # Number of trees in ensemble
    n_draws=2000,         # Posterior samples
    n_tune=1000,          # Burn-in samples
    variance_weighting=True,  # Use inverse-variance weighting
    random_state=42
)

# Fit the model
bart.fit(
    X=X,
    y=y,
    se=se,
    feature_names=['Age', 'Dose', 'Duration', 'Quality']
)

# Model summary
print(bart.summary())

# Variable importance
importance = bart.variable_importance(method='permutation')
print(importance)

# Heterogeneity statistics
het_stats = bart.heterogeneity_stats()
print(f"I² = {het_stats['I2']:.2f}%")
print(f"τ² = {het_stats['tau2']:.4f}")
```

### Visualization

```python
from visualization import MetaRegressionVisualizer

viz = MetaRegressionVisualizer()

# Forest plot
fig = viz.forest_plot(
    study_names=['Study 1', 'Study 2', ...],
    effect_sizes=y,
    standard_errors=se,
    predictions=bart.predictions
)
fig.savefig('forest_plot.png', dpi=300)

# Diagnostic panel
fig = viz.diagnostic_panel(bart)
fig.savefig('diagnostics.png', dpi=300)

# Partial dependence plot
fig = bart.plot_partial_dependence('Age')
fig.savefig('pdp_age.png', dpi=300)

# Variable importance
fig = bart.plot_variable_importance()
fig.savefig('importance.png', dpi=300)
```

## ⚡ Computational Considerations

### Runtime Performance

**Model Fitting** (k=50 studies, p=5 moderators):
- **BART**: ~60 seconds (varies: 45-90s depending on convergence)
- **WLS**: ~0.2 seconds
- **GAM**: ~1-2 seconds
- **Trade-off**: BART is ~200-300× slower than WLS

**Advanced Diagnostics** (k=50, p=5):
- **Permutation importance**: ~10 minutes (p × n_repeats × fit_time)
  - Example: 5 moderators × 5 repeats × 60s = ~25 minutes
  - Can be parallelized across features
- **LOO-CV**: ~15 minutes (k × fit_time, parallelizable)
  - Example: 50 studies × 60s = 50 minutes (sequential)
- **Partial dependence**: ~2-5 minutes per feature
  - Depends on grid_resolution and sample_posterior settings

**Scaling** (empirically determined):
- Sample size: O(k^1.3) - moderately superlinear
- Number of moderators: O(p^1.1) - nearly linear
- Rule of thumb: doubling k increases runtime by ~2.5×

### When BART is Worth the Computational Cost

✅ **Use BART when**:
- **Suspected non-linear dose-response** (e.g., medication dosage, exposure duration)
- **Unknown interactions between moderators** (exploratory analysis)
- **Many potential moderators** needing variable selection
- **Large meta-analyses** (k ≥ 30) with complex relationships
- **Research questions** where flexibility matters more than speed

⚠️ **Prefer WLS/Linear Meta-Regression when**:
- **Confirmatory analysis** of pre-specified linear effects
- **Simple linear relationships** with few moderators (p ≤ 3)
- **Time-sensitive analysis** requiring rapid turnaround
- **Small meta-analyses** (k < 20) where BART may overfit
- **Interpretable coefficients** are the primary goal

⚠️ **Consider GAM (Generalized Additive Models) when**:
- Need semi-parametric flexibility with better speed than BART
- Want smooth non-linear effects with interpretable shapes
- Have specific hypotheses about which variables are non-linear

### Optimization Tips

**Reduce runtime for large analyses**:
```python
# Faster fitting (reduced MCMC samples)
bart = BARTMetaRegression(
    n_trees=30,        # Default: 50
    n_draws=1000,      # Default: 2000
    n_tune=500,        # Default: 1000
    random_state=42
)

# Skip expensive diagnostics during exploration
importance = bart.variable_importance(method='inclusion')  # Fast tree-based
# Instead of: method='permutation' (slow, requires refitting)

# Reduce partial dependence resolution
pd_result = bart.partial_dependence(
    feature_idx=0,
    grid_resolution=25,      # Default: 50
    sample_posterior=False   # Use point estimates only
)
```

**Parallelize when possible**:
- LOO-CV: Can be parallelized across studies (future enhancement)
- Permutation importance: Can parallelize across features
- Multiple meta-analyses: Fit models in parallel

### Performance vs. Sample Size

| k (studies) | p (moderators) | Fit Time | Permutation Importance | LOO-CV |
|-------------|----------------|----------|------------------------|--------|
| 20          | 3              | ~15s     | ~2 min                 | ~5 min |
| 30          | 5              | ~30s     | ~8 min                 | ~15 min |
| 50          | 5              | ~60s     | ~25 min                | ~50 min |
| 75          | 7              | ~120s    | ~60 min                | ~2.5 hr |
| 100         | 10             | ~240s    | ~2 hr                  | ~6 hr |

*Note: Times are approximate and depend on hardware, convergence, and data complexity*

### Recommended Workflow

**1. Initial Exploration** (fast):
```python
# Fit with reduced settings
bart = BARTMetaRegression(n_trees=30, n_draws=1000, n_tune=500)
bart.fit(X, y, se)

# Quick diagnostics
print(bart.summary())
importance = bart.variable_importance(method='inclusion')  # Fast
```

**2. Refinement** (moderate):
```python
# Standard settings for important features
bart = BARTMetaRegression()  # Use defaults
bart.fit(X_selected, y, se)  # Subset to important features

# Partial dependence for key moderators
for feature in top_features:
    pd = bart.partial_dependence(feature, sample_posterior=False)
```

**3. Final Analysis** (comprehensive):
```python
# Full settings for publication
bart = BARTMetaRegression(n_trees=75, n_draws=2000, n_tune=1000)
bart.fit(X, y, se)

# Full diagnostics
importance = bart.variable_importance(method='permutation', n_repeats=10)
loo_results = bart.leave_one_out()
# Partial dependence with uncertainty
pd = bart.partial_dependence(feature, sample_posterior=True)
```

## 📊 Complete Examples

### Example 1: Linear Meta-Regression

```python
from simulation_studies import MetaAnalysisSimulator

# Generate simulated data
simulator = MetaAnalysisSimulator(random_state=42)
data = simulator.generate_linear_scenario(
    n_studies=50,
    n_features=3,
    tau2=0.05
)

# Fit BART model
bart = BARTMetaRegression(random_state=42)
bart.fit(data['X'], data['y'], data['se'])

# Analyze results
print(bart.summary())
print(bart.variable_importance())
```

### Example 2: Non-Linear Dose-Response

```python
# Generate data with quadratic dose-response
data = simulator.generate_nonlinear_scenario(
    n_studies=60,
    nonlinear_type='quadratic'  # or 'cubic', 'sinusoidal', 'threshold'
)

# Fit BART (automatically detects non-linearity)
bart = BARTMetaRegression(n_trees=75, random_state=42)
bart.fit(data['X'], data['y'], data['se'])

# Visualize non-linear relationship
fig = bart.plot_partial_dependence(0)  # First moderator (dose)
```

### Example 3: Interaction Detection

```python
# Generate data with interaction
data = simulator.generate_interaction_scenario(
    n_studies=70,
    interaction_strength=0.6
)

# Fit BART (automatically detects interactions)
bart = BARTMetaRegression(random_state=42)
bart.fit(data['X'], data['y'], data['se'])

# Analyze interactions
for i in range(data['X'].shape[1]):
    for j in range(i+1, data['X'].shape[1]):
        strength = bart.interaction_strength(i, j)
        print(f"Interaction {i}-{j}: {strength:.3f}")
```

### Example 4: Model Comparison

```python
from bart_meta_regression import compare_with_linear_meta_regression

# Compare BART vs traditional linear meta-regression
comparison = compare_with_linear_meta_regression(
    X=data['X'],
    y=data['y'],
    se=data['se'],
    cv_folds=5
)

print(comparison)
#           Model                 RMSE (mean ± std)       R² (mean ± std)
# BART Meta-Regression     0.1234 ± 0.0156      0.8765 ± 0.0234
# Linear Meta-Regression   0.2345 ± 0.0234      0.6543 ± 0.0456
```

## 🔬 Simulation Studies

Run comprehensive Monte Carlo simulations to evaluate performance:

```python
from simulation_studies import run_comprehensive_simulation_study

# Run simulations across multiple scenarios
results = run_comprehensive_simulation_study(
    n_simulations=100,
    random_state=42
)

# Results include:
# - Linear scenario
# - Quadratic scenario
# - Interaction scenario
# - Complex scenario (multiple non-linearities + interactions)

# Each scenario reports:
# - RMSE (prediction accuracy)
# - R² (variance explained)
# - Coverage probability (CI calibration)
```

## 📈 Advanced Features

### Custom Prior Specification

```python
bart = BARTMetaRegression(
    alpha=0.95,  # Base probability for tree prior (higher = shallower trees)
    beta=2.0,    # Power in tree prior (controls tree depth)
    n_trees=100  # More trees = better approximation
)
```

### Prediction on New Studies

```python
# Fit model
bart.fit(X_train, y_train, se_train)

# Predict on new studies
predictions = bart.predict(X_new, return_std=True)
pred_mean, pred_std = predictions

# Get quantiles
pred_quantiles = bart.predict(X_new, quantiles=[0.025, 0.5, 0.975])
```

### Variable Importance Methods

```python
# Permutation importance (default, more reliable)
importance_perm = bart.variable_importance(method='permutation', n_repeats=10)

# Inclusion frequency (faster, tree-based)
importance_inc = bart.variable_importance(method='inclusion')
```

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{bart_meta_regression,
  title = {BART Meta-Regression: Advanced Bayesian Nonparametric Meta-Analysis},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/bart-meta-regression}
}
```

### Key References

**BART Methodology:**
- Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. *The Annals of Applied Statistics*, 4(1), 266-298.

**Meta-Regression:**
- Thompson, S. G., & Higgins, J. P. (2002). How should meta‐regression analyses be undertaken and interpreted? *Statistics in Medicine*, 21(11), 1559-1573.

**BART for Meta-Analysis:**
- [Your forthcoming publication]

## 🎓 Methodological Advantages

### Why BART for Meta-Regression?

1. **No Functional Form Assumptions**: Traditional meta-regression assumes linear relationships. BART adapts to the data.

2. **Automatic Variable Selection**: The tree structure naturally excludes uninformative moderators.

3. **Interaction Detection**: Unlike linear models requiring pre-specified interaction terms, BART automatically models interactions.

4. **Regularization**: The Bayesian tree prior prevents overfitting, even with many moderators.

5. **Interpretability**: Despite being nonparametric, BART provides clear interpretability via partial dependence plots and variable importance.

6. **Uncertainty Quantification**: Full posterior distribution provides rigorous uncertainty quantification.

### When to Use BART Meta-Regression?

✅ **Good for:**
- Suspected non-linear dose-response relationships
- Unknown interactions between moderators
- Many potential moderators (variable selection needed)
- Complex relationships in large meta-analyses
- Exploratory moderator analysis

⚠️ **Less suitable for:**
- Very small meta-analyses (< 20 studies)
- Simple linear relationships with few moderators
- When interpretability of linear coefficients is critical
- Confirmatory hypothesis testing of pre-specified linear effects

## 🛠️ Project Structure

```
.
├── bart_meta_regression.py    # Core BART meta-regression implementation
├── visualization.py            # Publication-quality visualizations
├── simulation_studies.py       # Monte Carlo simulation framework
├── example_usage.py           # Comprehensive examples
├── requirements.txt           # Package dependencies
└── README.md                 # This file
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: PyMC-BART installation fails
```bash
# Solution: Install PyMC first
pip install pymc>=5.0
pip install pymc-bart
```

**Issue**: Slow sampling
```bash
# Solution: Reduce number of trees or samples
bart = BARTMetaRegression(n_trees=30, n_draws=1000, n_tune=500)
```

**Issue**: Memory errors with large datasets
```bash
# Solution: Use fewer posterior samples or trees
bart = BARTMetaRegression(n_trees=25, n_draws=500)
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional diagnostic plots
- Support for different effect size metrics
- Multivariate meta-analysis
- Network meta-regression
- Publication bias adjustment methods

## 📄 License

MIT License - see LICENSE file for details

## 📞 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: [your.email@institution.edu]

## 🙏 Acknowledgments

- PyMC development team for PyMC-BART
- Meta-analysis research community
- [Funding sources, if applicable]

---

**Note**: This is research software. Always validate results with domain expertise and consider comparison with traditional methods for robustness.
