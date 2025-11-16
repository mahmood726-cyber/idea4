# Manuscript Template: BART Meta-Regression

## Title Suggestions

1. "Bayesian Additive Regression Trees for Meta-Regression: A Flexible Alternative to Linear Meta-Regression"
2. "Nonparametric Meta-Regression Using BART: Automatic Detection of Non-Linear Relationships and Interactions"
3. "BART Meta-Regression: Advancing Meta-Analytic Methodology Through Machine Learning"

## Abstract Template (250 words)

**Background**: Traditional meta-regression assumes linear relationships between study characteristics and effect sizes, which may be overly restrictive. [State specific problem in your field].

**Objective**: To introduce and evaluate Bayesian Additive Regression Trees (BART) as a flexible, nonparametric alternative for meta-regression analysis.

**Methods**: BART meta-regression models effect sizes as a sum of regression trees, naturally accommodating non-linear relationships, interactions, and automatic variable selection. We conducted Monte Carlo simulations (N=[X] iterations) across four scenarios: linear, quadratic, interaction, and complex multi-moderator patterns. We compared BART with traditional weighted linear meta-regression using root mean squared error (RMSE), R², and 95% confidence interval coverage. We demonstrate the method using [describe your application dataset].

**Results**: In simulation studies, BART outperformed linear meta-regression in scenarios with non-linear relationships (RMSE: [X] vs [Y], p<0.001) and interactions ([results]). Coverage probabilities remained near nominal levels (94-96%). BART correctly identified important moderators while excluding noise variables. In the empirical application to [your dataset] ([K] studies, N=[total N]), BART revealed [key substantive findings, e.g., inverted-U dose-response, interaction between X and Y].

**Conclusions**: BART meta-regression provides a powerful, interpretable framework for exploring complex moderator effects. The method is particularly valuable when relationships are suspected to be non-linear, multiple interactions are plausible, or automatic variable selection is needed.

**Keywords**: meta-analysis, meta-regression, BART, Bayesian methods, nonparametric regression, machine learning

---

## Introduction

### Paragraph 1: Meta-Analysis Context
- Importance of meta-analysis in synthesizing evidence
- Role of meta-regression in explaining heterogeneity
- Your specific research area and why meta-regression is important

### Paragraph 2: Limitations of Linear Meta-Regression
- Assumption of linear relationships
- Need to pre-specify interactions
- Overfitting risk with many moderators
- Difficulty with variable selection

### Paragraph 3: BART as Solution
- Brief introduction to BART (Chipman et al., 2010)
- Key advantages: nonparametric, automatic interaction detection, regularization
- Growing use in other fields (cite applications)
- Gap: Limited application to meta-analysis

### Paragraph 4: Study Objectives
1. Introduce BART meta-regression methodology
2. Evaluate performance via simulation studies
3. Demonstrate application to [your research question]
4. Provide software implementation

---

## Methods

### 2.1 Traditional Meta-Regression

**Fixed-Effect Model**:
```
θᵢ = β₀ + Σⱼ βⱼXⱼᵢ + εᵢ
εᵢ ~ N(0, σᵢ²)
```

**Random-Effects Model**:
```
θᵢ = β₀ + Σⱼ βⱼXⱼᵢ + uᵢ + εᵢ
uᵢ ~ N(0, τ²)
εᵢ ~ N(0, σᵢ²)
```

- Describe estimation (weighted least squares)
- Discuss limitations (linearity, pre-specified interactions)

### 2.2 BART Meta-Regression

**Model Specification**:
```
yᵢ ~ N(f(Xᵢ), σᵢ²)
f(X) = Σₘ g(X; Tₘ, Mₘ)
```

where:
- yᵢ = observed effect size for study i
- f(Xᵢ) = sum of M regression trees
- g(X; Tₘ, Mₘ) = function defined by tree m
- σᵢ² = known within-study variance

**Prior Specification**:
- Tree structure prior: P(Tₘ) ∝ α(1 + d)^(-β)
- Terminal node parameters: Mⱼ ~ N(μ_μ, σ²_μ)
- Variance weighting: Studies weighted by 1/σᵢ²

**Posterior Inference**:
- Backfitting MCMC algorithm
- [N_draws] posterior samples after [N_tune] burn-in
- Convergence diagnostics: [specify]

### 2.3 Variable Importance

**Permutation Importance**:
1. Calculate baseline prediction error
2. Permute each variable independently
3. Recalculate prediction error
4. Importance = increase in error

**Inclusion Frequency**:
- Proportion of trees splitting on variable
- Weighted by split depth

### 2.4 Partial Dependence

For variable Xⱼ:
```
f̂ⱼ(xⱼ) = (1/n) Σᵢ f̂(xⱼ, X₋ⱼ,ᵢ)
```

- Shows marginal effect of Xⱼ averaging over other variables
- Reveals non-linear relationships
- Uncertainty via posterior distribution

### 2.5 Interaction Detection

**Friedman's H-statistic**:
- Measures departure from additivity
- H = 0: no interaction
- H = 1: complete interaction

### 2.6 Simulation Design

**Scenarios**:
1. **Linear**: Linear moderator effects (baseline)
2. **Quadratic**: X₁ has inverted-U relationship
3. **Interaction**: X₁ × X₂ interaction
4. **Complex**: Multiple non-linearities + interactions + noise variables

**Data Generation**:
- K = [40, 60, 80, 100] studies
- p = [3, 4, 6, 8] moderators
- τ² = [0.04, 0.06, 0.08] heterogeneity levels
- Sample sizes: N ~ N(100, 30), minimum 20

**Evaluation Metrics**:
1. RMSE: Root mean squared prediction error
2. R²: Variance explained
3. Coverage: 95% CI coverage of true effects
4. Variable selection accuracy (for complex scenario)

**Monte Carlo**: N = [50-100] iterations per scenario

### 2.7 Empirical Application

**Dataset**: [Describe your dataset]
- Search strategy
- Inclusion/exclusion criteria
- Effect size metric (e.g., log OR, SMD)
- Moderators examined
- K studies, total N

**Analysis Plan**:
1. Fit BART meta-regression
2. Assess heterogeneity (I², τ²)
3. Identify important moderators
4. Examine partial dependence plots
5. Compare with linear meta-regression
6. Sensitivity analyses

### 2.8 Software

- Python 3.8+ with PyMC-BART
- Code available at: [GitHub URL]
- Analysis reproducible via provided scripts

---

## Results

### 3.1 Simulation Study

#### Table 1: Simulation Results by Scenario

| Scenario | Method | RMSE (SD) | R² (SD) | Coverage (%) |
|----------|--------|-----------|---------|--------------|
| Linear   | BART   | X.XX (X.XX) | X.XX (X.XX) | XX.X |
|          | Linear | X.XX (X.XX) | X.XX (X.XX) | XX.X |
| Quadratic| BART   | X.XX (X.XX) | X.XX (X.XX) | XX.X |
|          | Linear | X.XX (X.XX) | X.XX (X.XX) | XX.X |
| Interaction | BART | X.XX (X.XX) | X.XX (X.XX) | XX.X |
|          | Linear | X.XX (X.XX) | X.XX (X.XX) | XX.X |
| Complex  | BART   | X.XX (X.XX) | X.XX (X.XX) | XX.X |
|          | Linear | X.XX (X.XX) | X.XX (X.XX) | XX.X |

**Key Findings**:
- BART matched linear performance in linear scenario
- BART substantially outperformed in non-linear scenarios
- Coverage probabilities near nominal 95%
- Variable selection correctly identified important moderators

#### Figure 1: Simulation Results
[Include: Violin plots of RMSE, R², Coverage across scenarios]

#### Figure 2: Variable Importance in Complex Scenario
[Include: Bar plot showing true vs estimated importance]

### 3.2 Empirical Application

**Dataset Characteristics**:
- K = [X] studies
- Total N = [X] participants
- Effect sizes ranged from [X] to [X]
- Median SE = [X]

**Heterogeneity Assessment**:
- Q = [X.XX], df = [X], p < 0.001
- I² = [X.X]% (95% CI: [X.X, X.X])
- τ² = [X.XXX]

#### Table 2: Model Comparison

| Model | R² | RMSE | Δ Heterogeneity Explained |
|-------|-----|------|---------------------------|
| Null (no moderators) | - | X.XX | - |
| Linear meta-regression | X.XX | X.XX | XX% |
| BART meta-regression | X.XX | X.XX | XX% |

BART explained additional [X]% of heterogeneity beyond linear model.

#### Table 3: Variable Importance Rankings

| Moderator | Importance | Rank |
|-----------|-----------|------|
| [Variable 1] | X.XXX | 1 |
| [Variable 2] | X.XXX | 2 |
| [Variable 3] | X.XXX | 3 |
| ... | ... | ... |

#### Figure 3: Forest Plot
[Include: Observed effects vs BART predictions with 95% CIs]

#### Figure 4: Partial Dependence Plots
[Include: PDPs for top 3-4 moderators showing relationships]

**Key Finding 1: Non-Linear Dose-Response**
- Partial dependence revealed inverted-U relationship for [moderator]
- Optimal effect at [value]
- Diminishing returns beyond [value]

**Key Finding 2: Interaction Effects**
- Strong interaction detected between [X] and [Y] (H = [X.XX])
- Effect of [X] depends on level of [Y]

**Key Finding 3: Variable Selection**
- [X] of [total] moderators showed importance > 0.01
- [List variables with negligible effects]

### 3.3 Diagnostic Analyses

#### Figure 5: Residual Diagnostics
[Include: Residuals vs fitted, Q-Q plot, scale-location]

#### Figure 6: Funnel Plot
[Include: Funnel plot for publication bias assessment]

**Publication Bias Assessment**:
- Egger's test: t = [X.XX], p = [X.XX]
- Funnel plot shows [symmetric/asymmetric] pattern
- [Interpretation]

### 3.4 Sensitivity Analyses

**Prior Sensitivity**:
- Results robust to α ∈ {0.90, 0.95, 0.99}
- [Specify any differences]

**Influential Studies**:
- [X] studies with |standardized residuals| > 3
- Results unchanged when excluding influential studies

---

## Discussion

### 4.1 Summary of Findings
- Recap main results
- BART successfully detected [specific non-linear patterns/interactions]
- Outperformed linear meta-regression by [metric]

### 4.2 Substantive Implications
- What do your specific findings mean for [your field]?
- Clinical/practical implications
- Policy implications

### 4.3 Methodological Advantages
1. **Automatic interaction detection** without pre-specification
2. **Non-linear modeling** captures complex dose-response
3. **Variable selection** handles many potential moderators
4. **Regularization** prevents overfitting
5. **Interpretability** via partial dependence plots

### 4.4 When to Use BART Meta-Regression

**Recommended when**:
- Non-linear relationships suspected
- Many potential moderators (p > 5)
- Interaction patterns unknown
- Adequate sample size (K ≥ 20-30 studies)
- Exploratory moderator analysis

**Less suitable when**:
- Small meta-analyses (K < 20)
- Simple linear relationships expected
- Confirmatory hypothesis testing
- Linear coefficients needed for interpretation

### 4.5 Comparison with Other Approaches

**vs Linear Meta-Regression**:
- More flexible but requires more data
- [Specific comparison from your results]

**vs GAMs (Generalized Additive Models)**:
- BART: Automatic interaction detection
- GAM: Smoother curves but interactions must be specified

**vs Meta-CART/Random Forests**:
- BART: Better uncertainty quantification (Bayesian)
- RF: No UQ, harder to interpret

### 4.6 Limitations

1. **Sample size**: Requires adequate studies (≥20-30)
2. **Computational cost**: Slower than linear regression
3. **Extrapolation**: Like all methods, limited to observed covariate range
4. **Interpretability**: Less direct than linear coefficients
5. **Software**: Requires familiarity with Bayesian methods

### 4.7 Future Directions

- Multivariate meta-analysis with BART
- Network meta-regression
- Individual patient data meta-analysis
- Incorporation of publication bias models
- Real-time updating as new studies emerge

### 4.8 Conclusions

BART meta-regression provides a powerful, flexible framework for exploring complex moderator effects in meta-analysis. The method naturally accommodates non-linear relationships and interactions while providing robust uncertainty quantification. Our simulation and empirical results demonstrate [key conclusions]. We provide open-source software to facilitate adoption.

---

## References

### Key BART References

Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. *The Annals of Applied Statistics*, 4(1), 266-298.

### Meta-Regression References

Borenstein, M., Hedges, L. V., Higgins, J. P., & Rothstein, H. R. (2009). *Introduction to meta-analysis*. John Wiley & Sons.

Thompson, S. G., & Higgins, J. P. (2002). How should meta‐regression analyses be undertaken and interpreted? *Statistics in Medicine*, 21(11), 1559-1573.

### [Add field-specific references]

---

## Tables

### Table 1: Simulation Design
[Detailed simulation parameters]

### Table 2: Dataset Characteristics
[Descriptive statistics of your empirical dataset]

### Table 3: Model Comparison
[BART vs Linear performance metrics]

### Table 4: Variable Importance
[Ranked moderator importance]

---

## Figures

### Figure 1: Conceptual Diagram
[Show how BART works vs linear regression]

### Figure 2: Simulation Results
[Performance across scenarios]

### Figure 3: Forest Plot
[Your empirical results]

### Figure 4: Partial Dependence Plots
[Key non-linear relationships]

### Figure 5: Interaction Effects
[If applicable, show interaction patterns]

### Figure 6: Diagnostics
[Residual plots, Q-Q plot]

### Figure 7: Model Comparison
[BART vs Linear performance]

---

## Supplementary Materials

### Supplement A: Detailed Methodology
- BART algorithm details
- Prior specification rationale
- MCMC convergence diagnostics

### Supplement B: Full Simulation Results
- All scenarios and parameter combinations
- Sensitivity analyses

### Supplement C: Complete Variable Importance
- All moderators examined
- Inclusion frequencies

### Supplement D: Software Tutorial
- Step-by-step guide
- Reproducible analysis code
- Example datasets

### Supplement E: Sensitivity Analyses
- Influential study analyses
- Prior sensitivity
- Alternative specifications

---

## Data Availability Statement

The data and code used in this study are available at: [GitHub repository URL]. The repository includes:
- All analysis scripts
- Simulation code
- Example datasets
- Complete documentation

---

## Author Contributions

[As appropriate for your authorship]

---

## Funding

[Acknowledge funding sources]

---

## Conflicts of Interest

The authors declare no conflicts of interest.

---

## Acknowledgments

We thank [individuals/organizations]. This work used PyMC-BART developed by [contributors].
