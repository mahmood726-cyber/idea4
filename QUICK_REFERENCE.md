# BART Meta-Regression: Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from bart_meta_regression import BARTMetaRegression

# Fit model
bart = BARTMetaRegression(n_trees=50, n_draws=2000, random_state=42)
bart.fit(X, y, se, feature_names=['Age', 'Dose', 'Duration'])

# Results
print(bart.summary())
print(bart.variable_importance())
```

## Key Functions

### Model Fitting

| Function | Description |
|----------|-------------|
| `bart.fit(X, y, se)` | Fit BART meta-regression model |
| `bart.predict(X_new)` | Predict effect sizes for new studies |
| `bart.summary()` | Get model summary statistics |

### Analysis

| Function | Description |
|----------|-------------|
| `bart.variable_importance(method='permutation')` | Calculate variable importance |
| `bart.heterogeneity_stats()` | Get I², τ², Q statistics |
| `bart.partial_dependence(feature_idx)` | Compute partial dependence |
| `bart.interaction_strength(i, j)` | Measure interaction strength |

### Visualization

| Function | Description |
|----------|-------------|
| `bart.plot_variable_importance()` | Plot variable importance |
| `bart.plot_partial_dependence(feature)` | Plot partial dependence |
| `bart.plot_residuals()` | Plot residual diagnostics |

### Advanced

| Function | Description |
|----------|-------------|
| `compare_with_linear_meta_regression()` | Compare BART vs linear |
| `MetaRegressionVisualizer.forest_plot()` | Create forest plot |
| `MetaRegressionVisualizer.diagnostic_panel()` | Full diagnostic panel |
| `MetaRegressionVisualizer.funnel_plot()` | Publication bias funnel plot |

## Common Workflows

### 1. Basic Analysis

```python
# Fit model
bart = BARTMetaRegression(random_state=42)
bart.fit(X, y, se, feature_names=names)

# Examine results
print(bart.summary())
importance = bart.variable_importance()
het_stats = bart.heterogeneity_stats()
```

### 2. Visualization

```python
from visualization import MetaRegressionVisualizer

viz = MetaRegressionVisualizer()

# Forest plot
viz.forest_plot(study_names, y, se, bart.predictions)

# Diagnostics
viz.diagnostic_panel(bart)

# Partial dependence
bart.plot_partial_dependence('Age')
```

### 3. Model Comparison

```python
from bart_meta_regression import compare_with_linear_meta_regression

comparison = compare_with_linear_meta_regression(X, y, se, cv_folds=5)
print(comparison)
```

### 4. Simulation Study

```python
from simulation_studies import run_comprehensive_simulation_study

results = run_comprehensive_simulation_study(n_simulations=50)
```

## Parameter Tuning

### BART Parameters

| Parameter | Default | Description | When to adjust |
|-----------|---------|-------------|----------------|
| `n_trees` | 50 | Number of trees | Increase for complex relationships (75-100) |
| `n_draws` | 2000 | Posterior samples | Increase for precision (3000-5000) |
| `n_tune` | 1000 | Burn-in samples | Increase if convergence issues (1500-2000) |
| `alpha` | 0.95 | Tree depth prior | Decrease for deeper trees (0.90-0.99) |
| `beta` | 2.0 | Depth power | Usually keep default |
| `variance_weighting` | True | Use inverse-variance weights | Always True for meta-analysis |

## Interpretation Guide

### Variable Importance

- **> 0.02**: Strong effect
- **0.01-0.02**: Moderate effect
- **0.005-0.01**: Weak effect
- **< 0.005**: Negligible effect

### Heterogeneity (I²)

- **0-30%**: Low heterogeneity
- **30-60%**: Moderate heterogeneity
- **60-75%**: Substantial heterogeneity
- **> 75%**: Considerable heterogeneity

### Model Fit (R²)

- **< 0.30**: Weak
- **0.30-0.50**: Moderate
- **0.50-0.70**: Good
- **> 0.70**: Strong

## Example Datasets

### Generate Simulated Data

```python
from simulation_studies import MetaAnalysisSimulator

sim = MetaAnalysisSimulator(random_state=42)

# Linear scenario
data_linear = sim.generate_linear_scenario(n_studies=50)

# Non-linear scenario
data_nonlinear = sim.generate_nonlinear_scenario(
    n_studies=60,
    nonlinear_type='quadratic'
)

# Interaction scenario
data_interaction = sim.generate_interaction_scenario(
    n_studies=70,
    interaction_strength=0.5
)

# Complex scenario
data_complex = sim.generate_complex_scenario(n_studies=100)
```

## Running Examples

### Quick Start (5 minutes)

```bash
python quickstart.py
```

### Comprehensive Examples (30 minutes)

```bash
python example_usage.py
```

### Individual Examples

```python
from example_usage import (
    example_1_linear_meta_regression,
    example_2_nonlinear_meta_regression,
    example_3_interaction_detection,
    example_4_model_comparison
)

example_1_linear_meta_regression()
```

## Troubleshooting

### Problem: Slow sampling

**Solution**: Reduce `n_trees`, `n_draws`, or `n_tune`

```python
bart = BARTMetaRegression(n_trees=30, n_draws=1000, n_tune=500)
```

### Problem: Poor convergence

**Solution**: Increase `n_tune` or adjust priors

```python
bart = BARTMetaRegression(n_tune=2000, alpha=0.90)
```

### Problem: Memory error

**Solution**: Reduce number of trees or samples

```python
bart = BARTMetaRegression(n_trees=25, n_draws=500)
```

### Problem: PyMC-BART not found

**Solution**: Install dependencies in order

```bash
pip install pymc>=5.0
pip install pymc-bart
```

## Publication Checklist

- [ ] Report n_trees, n_draws, n_tune
- [ ] Include heterogeneity statistics (I², τ², Q)
- [ ] Show variable importance ranking
- [ ] Present model fit (R², RMSE)
- [ ] Compare with linear meta-regression
- [ ] Include diagnostic plots
- [ ] Report partial dependence for key moderators
- [ ] Provide forest plot
- [ ] Assess publication bias (funnel plot)

## Citing This Software

```bibtex
@software{bart_meta_regression2025,
  title = {BART Meta-Regression: Advanced Bayesian Nonparametric Meta-Analysis},
  year = {2025},
  url = {https://github.com/mahmood726-cyber/idea4}
}
```

## Additional Resources

- **Full Documentation**: See `README.md`
- **Tutorial**: See `tutorial.md`
- **Examples**: Run `example_usage.py`
- **Utilities**: See `utils.py` for helper functions

## Contact

- GitHub Issues: https://github.com/mahmood726-cyber/idea4/issues
- Email: your.email@institution.edu

---

**Version**: 1.0.0
**Last Updated**: 2025-01-16
