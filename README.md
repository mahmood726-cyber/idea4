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
- Leave-one-out cross-validation (with parallelization support)
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
# Install all requirements
pip install -r requirements.txt

# Or install core dependencies individually
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install pymc>=5.0 pymc-bart>=0.5 arviz>=0.16
pip install statsmodels joblib pytest
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
    estimate_tau=True,    # Estimate between-study heterogeneity
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
print(f"I² = {het_stats['I2_bart']:.2f}%")
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
    predictions=bart.predictions_mean
)
fig.savefig('forest_plot.png', dpi=300)

# Diagnostic panel
fig = viz.diagnostic_panel(bart)
fig.savefig('diagnostics.png', dpi=300)

# Partial dependence plot
fig = bart.plot_partial_dependence(0)  # First moderator
fig.savefig('pdp.png', dpi=300)

# Variable importance
fig = bart.plot_variable_importance()
fig.savefig('importance.png', dpi=300)
```

## 🎓 When to Use BART Meta-Regression?

### ✅ BART is Well-Suited For:

- **Suspected non-linear dose-response** (e.g., medication dosage, exposure duration)
- **Unknown interactions between moderators** (exploratory analysis)
- **Many potential moderators** needing variable selection
- **Large meta-analyses** (k ≥ 30) with complex relationships
- **Research questions** where flexibility matters more than speed

### ⚠️ Prefer Traditional Linear Meta-Regression When:

- **Confirmatory analysis** of pre-specified linear effects
- **Simple linear relationships** with few moderators (p ≤ 3)
- **Time-sensitive analysis** requiring rapid turnaround
- **Small meta-analyses** (k < 20) where BART may overfit
- **Interpretable coefficients** are the primary goal

### Sample Size Recommendations

| Heterogeneity (I²) | Minimum Studies (k) | Recommendation |
|--------------------|---------------------|----------------|
| Low (< 25%)        | ≥ 20                | May suffice    |
| Moderate (25-75%)  | ≥ 30                | Recommended    |
| High (> 75%)       | ≥ 50                | Recommended    |
| k < 20             | -                   | Use WLS instead|

## 📚 Documentation

This project includes comprehensive documentation:

- **README.md** (this file) - Quick start and overview
- **[TUTORIAL.md](TUTORIAL.md)** - Step-by-step examples and complete workflows
- **[ADVANCED.md](ADVANCED.md)** - Computational details, performance tuning, and customization
- **[tutorial.md](tutorial.md)** - Original detailed tutorial (comprehensive reference)

## 🔬 Example Workflows

### Simple Analysis

```python
from bart_meta_regression import BARTMetaRegression
from simulation_studies import MetaAnalysisSimulator

# Generate simulated data
simulator = MetaAnalysisSimulator(random_state=42)
data = simulator.generate_linear_scenario(n_studies=50, n_features=3)

# Fit BART model
bart = BARTMetaRegression(random_state=42)
bart.fit(data['X'], data['y'], data['se'])

# Analyze results
print(bart.summary())
importance = bart.variable_importance(method='inclusion')  # Fast method
```

### Advanced Analysis with Diagnostics

```python
# Fit with full settings
bart = BARTMetaRegression(
    n_trees=75,
    n_draws=2000,
    n_tune=1000,
    estimate_tau=True,
    random_state=42
)
bart.fit(X, y, se, feature_names=feature_names)

# Comprehensive diagnostics
importance = bart.variable_importance(method='permutation', n_repeats=10)
loo_results = bart.leave_one_out(verbose=True, n_jobs=-1)  # Parallel LOO-CV
het_stats = bart.heterogeneity_stats()

# Partial dependence for top moderators
for feature_idx in range(3):
    fig = bart.plot_partial_dependence(
        feature_idx,
        grid_resolution=50,
        sample_posterior=True
    )
    fig.savefig(f'pdp_feature_{feature_idx}.png', dpi=300)
```

See **[TUTORIAL.md](TUTORIAL.md)** for complete step-by-step examples.

## ⚡ Performance Notes

**Typical Runtime** (k=50 studies, p=5 moderators):
- Model fitting: ~60 seconds
- Permutation importance: ~10-25 minutes
- LOO-CV (sequential): ~50 minutes
- LOO-CV (parallel, 8 cores): ~6-8 minutes

**NEW in v2.2.0**: LOO-CV now supports parallelization for 3-15× speedup!

```python
# Use all CPU cores for LOO-CV
loo_results = bart.leave_one_out(verbose=True, n_jobs=-1)
```

See **[ADVANCED.md](ADVANCED.md)** for detailed performance analysis and optimization strategies.

## 🧪 Testing

This project uses pytest for comprehensive testing:

```bash
# Run all tests
pytest -v

# Run tests excluding slow ones
pytest -v -m "not slow"

# Run with coverage
pytest --cov=bart_meta_regression --cov-report=html
```

## 📊 Real Data Examples

The project includes a real data vignette using the classic BCG vaccine meta-analysis:

```python
# See bcg_vaccine_vignette.py for complete example
from bcg_vaccine_vignette import run_bcg_analysis

results = run_bcg_analysis()
```

This demonstrates BART meta-regression on a published meta-analysis dataset.

## 📈 Methodological Advantages

### Why BART for Meta-Regression?

1. **No Functional Form Assumptions**: Traditional meta-regression assumes linear relationships. BART adapts to the data.

2. **Automatic Variable Selection**: The tree structure naturally excludes uninformative moderators.

3. **Interaction Detection**: Unlike linear models requiring pre-specified interaction terms, BART automatically models interactions.

4. **Regularization**: The Bayesian tree prior prevents overfitting, even with many moderators.

5. **Interpretability**: Despite being nonparametric, BART provides clear interpretability via partial dependence plots and variable importance.

6. **Uncertainty Quantification**: Full posterior distribution provides rigorous uncertainty quantification.

## 🔗 Key References

**BART Methodology:**
- Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. *The Annals of Applied Statistics*, 4(1), 266-298.

**Meta-Regression:**
- Thompson, S. G., & Higgins, J. P. (2002). How should meta‐regression analyses be undertaken and interpreted? *Statistics in Medicine*, 21(11), 1559-1573.

**BCG Vaccine Data:**
- Colditz, G. A., et al. (1994). Efficacy of BCG vaccine in the prevention of tuberculosis. *JAMA*, 271(9), 698-702.

## 📄 Citation

If you use this software in your research, please cite:

```bibtex
@software{bart_meta_regression,
  title = {BART Meta-Regression: Advanced Bayesian Nonparametric Meta-Analysis},
  author = {Advanced Meta-Analysis Research Team},
  year = {2025},
  version = {2.2.0},
  url = {https://github.com/yourusername/bart-meta-regression}
}
```

## 🛠️ Project Structure

```
.
├── bart_meta_regression.py       # Core BART meta-regression implementation
├── visualization.py               # Publication-quality visualizations
├── simulation_studies.py          # Monte Carlo simulation framework
├── utils.py                       # Helper functions
├── example_usage.py              # Comprehensive examples
├── bcg_vaccine_vignette.py       # Real data analysis example
├── test_bart_fixes_pytest.py     # Pytest test suite
├── requirements.txt              # Package dependencies
├── pytest.ini                    # Pytest configuration
├── README.md                     # This file
├── TUTORIAL.md                   # Step-by-step tutorial
├── ADVANCED.md                   # Advanced usage and performance
└── data/                         # Example datasets
    └── bcg_vaccine.csv
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional diagnostic plots
- Support for different effect size metrics
- Multivariate meta-analysis
- Network meta-regression
- Publication bias adjustment methods

## 🐛 Troubleshooting

### Common Issues

**PyMC-BART installation fails:**
```bash
pip install pymc>=5.0
pip install pymc-bart>=0.5
```

**Slow sampling:**
```python
# Reduce number of trees or samples
bart = BARTMetaRegression(n_trees=30, n_draws=1000, n_tune=500)
```

**Memory errors:**
```python
# Use fewer posterior samples or trees
bart = BARTMetaRegression(n_trees=25, n_draws=500)
```

See **[ADVANCED.md](ADVANCED.md)** for detailed troubleshooting and optimization.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- PyMC development team for PyMC-BART
- Meta-analysis research community
- Peer reviewers who provided invaluable feedback

---

**Version**: 2.2.0 (Post-Acceptance Enhancements)

**Status**: Research software. Always validate results with domain expertise and consider comparison with traditional methods for robustness.

For detailed examples, see **[TUTORIAL.md](TUTORIAL.md)**. For performance optimization and advanced features, see **[ADVANCED.md](ADVANCED.md)**.
