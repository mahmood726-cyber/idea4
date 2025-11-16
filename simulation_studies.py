"""
Simulation Studies for BART Meta-Regression

Comprehensive simulation framework to demonstrate BART's advantages:
1. Non-linear relationship detection
2. Interaction modeling without pre-specification
3. Performance comparison with traditional methods
4. Power and Type I error analysis
5. Coverage probability assessment
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class MetaAnalysisSimulator:
    """
    Generate simulated meta-analytic data with known properties.

    This class creates realistic meta-analytic datasets with:
    - Customizable effect size distributions
    - Non-linear moderator effects
    - Interaction effects
    - Between-study heterogeneity
    - Publication bias (optional)
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize simulator.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def generate_linear_scenario(
        self,
        n_studies: int = 50,
        n_features: int = 3,
        beta_true: Optional[np.ndarray] = None,
        tau2: float = 0.05,
        mean_n: int = 100,
        sd_n: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Generate data with linear moderator effects.

        Parameters
        ----------
        n_studies : int, default=50
            Number of studies
        n_features : int, default=3
            Number of moderator variables
        beta_true : ndarray, optional
            True regression coefficients. If None, randomly generated.
        tau2 : float, default=0.05
            Between-study variance (heterogeneity)
        mean_n : int, default=100
            Mean sample size per study
        sd_n : int, default=30
            SD of sample size across studies

        Returns
        -------
        data : dict
            Dictionary containing:
            - 'X': Feature matrix
            - 'y': Effect sizes
            - 'se': Standard errors
            - 'beta_true': True coefficients
            - 'tau2': True between-study variance
            - 'study_names': Study identifiers
        """
        # Generate moderators
        X = np.random.randn(n_studies, n_features)

        # Generate true coefficients
        if beta_true is None:
            beta_true = np.random.randn(n_features) * 0.5
            beta_true[0] = 0.5  # Ensure at least one moderate effect

        # True mean effect (linear combination)
        theta_true = X @ beta_true

        # Add between-study heterogeneity
        tau = np.sqrt(tau2)
        theta = theta_true + np.random.randn(n_studies) * tau

        # Generate sample sizes
        n_per_study = np.maximum(20, np.random.normal(mean_n, sd_n, n_studies).astype(int))

        # Generate observed effect sizes with sampling error
        se = np.sqrt(4 / n_per_study)  # Approximate SE for SMD
        y = theta + np.random.randn(n_studies) * se

        study_names = [f"Study_{i+1}" for i in range(n_studies)]

        return {
            'X': X,
            'y': y,
            'se': se,
            'theta_true': theta_true,
            'beta_true': beta_true,
            'tau2': tau2,
            'study_names': study_names,
            'scenario': 'linear'
        }

    def generate_nonlinear_scenario(
        self,
        n_studies: int = 50,
        n_features: int = 3,
        tau2: float = 0.05,
        mean_n: int = 100,
        sd_n: int = 30,
        nonlinear_type: str = 'quadratic'
    ) -> Dict[str, np.ndarray]:
        """
        Generate data with non-linear moderator effects.

        Parameters
        ----------
        n_studies : int, default=50
            Number of studies
        n_features : int, default=3
            Number of moderator variables
        tau2 : float, default=0.05
            Between-study variance
        mean_n : int, default=100
            Mean sample size per study
        sd_n : int, default=30
            SD of sample size
        nonlinear_type : str, default='quadratic'
            Type of non-linearity: 'quadratic', 'cubic', 'sinusoidal', 'threshold'

        Returns
        -------
        data : dict
            Dictionary with same structure as generate_linear_scenario
        """
        # Generate moderators
        X = np.random.randn(n_studies, n_features)

        # Define non-linear relationship
        if nonlinear_type == 'quadratic':
            # Quadratic effect of first moderator
            theta_true = 0.3 + 0.5 * X[:, 0] - 0.3 * X[:, 0]**2
            if n_features > 1:
                theta_true += 0.2 * X[:, 1]
            if n_features > 2:
                theta_true += 0.15 * X[:, 2]

        elif nonlinear_type == 'cubic':
            # Cubic effect
            theta_true = 0.3 + 0.4 * X[:, 0] - 0.2 * X[:, 0]**2 + 0.1 * X[:, 0]**3
            if n_features > 1:
                theta_true += 0.2 * X[:, 1]

        elif nonlinear_type == 'sinusoidal':
            # Sinusoidal effect
            theta_true = 0.3 + 0.5 * np.sin(2 * X[:, 0])
            if n_features > 1:
                theta_true += 0.2 * X[:, 1]

        elif nonlinear_type == 'threshold':
            # Threshold effect
            theta_true = np.where(X[:, 0] > 0, 0.8, 0.2)
            if n_features > 1:
                theta_true += 0.2 * X[:, 1]

        else:
            raise ValueError(f"Unknown nonlinear_type: {nonlinear_type}")

        # Add between-study heterogeneity
        tau = np.sqrt(tau2)
        theta = theta_true + np.random.randn(n_studies) * tau

        # Generate sample sizes and observed effects
        n_per_study = np.maximum(20, np.random.normal(mean_n, sd_n, n_studies).astype(int))
        se = np.sqrt(4 / n_per_study)
        y = theta + np.random.randn(n_studies) * se

        study_names = [f"Study_{i+1}" for i in range(n_studies)]

        return {
            'X': X,
            'y': y,
            'se': se,
            'theta_true': theta_true,
            'tau2': tau2,
            'study_names': study_names,
            'scenario': f'nonlinear_{nonlinear_type}'
        }

    def generate_interaction_scenario(
        self,
        n_studies: int = 50,
        n_features: int = 4,
        tau2: float = 0.05,
        mean_n: int = 100,
        sd_n: int = 30,
        interaction_strength: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Generate data with interaction effects.

        Parameters
        ----------
        n_studies : int, default=50
            Number of studies
        n_features : int, default=4
            Number of moderator variables (minimum 2 for interaction)
        tau2 : float, default=0.05
            Between-study variance
        mean_n : int, default=100
            Mean sample size per study
        sd_n : int, default=30
            SD of sample size
        interaction_strength : float, default=0.5
            Strength of interaction effect

        Returns
        -------
        data : dict
            Dictionary with same structure as generate_linear_scenario
        """
        assert n_features >= 2, "Need at least 2 features for interaction"

        # Generate moderators
        X = np.random.randn(n_studies, n_features)

        # Main effects + interaction
        theta_true = (
            0.3 +  # Intercept
            0.4 * X[:, 0] +  # Main effect of X1
            0.3 * X[:, 1] +  # Main effect of X2
            interaction_strength * X[:, 0] * X[:, 1]  # Interaction
        )

        # Add other features if present
        for i in range(2, n_features):
            theta_true += 0.15 * X[:, i]

        # Add between-study heterogeneity
        tau = np.sqrt(tau2)
        theta = theta_true + np.random.randn(n_studies) * tau

        # Generate sample sizes and observed effects
        n_per_study = np.maximum(20, np.random.normal(mean_n, sd_n, n_studies).astype(int))
        se = np.sqrt(4 / n_per_study)
        y = theta + np.random.randn(n_studies) * se

        study_names = [f"Study_{i+1}" for i in range(n_studies)]

        return {
            'X': X,
            'y': y,
            'se': se,
            'theta_true': theta_true,
            'tau2': tau2,
            'study_names': study_names,
            'scenario': 'interaction',
            'interaction_strength': interaction_strength
        }

    def generate_complex_scenario(
        self,
        n_studies: int = 100,
        n_features: int = 8,
        tau2: float = 0.08,
        mean_n: int = 100,
        sd_n: int = 40
    ) -> Dict[str, np.ndarray]:
        """
        Generate complex scenario with multiple non-linearities and interactions.

        This represents a realistic meta-regression scenario with:
        - Non-linear effects
        - Interaction effects
        - Irrelevant variables (for testing variable selection)
        - Moderate heterogeneity

        Parameters
        ----------
        n_studies : int, default=100
            Number of studies
        n_features : int, default=8
            Number of moderator variables
        tau2 : float, default=0.08
            Between-study variance
        mean_n : int, default=100
            Mean sample size
        sd_n : int, default=40
            SD of sample size

        Returns
        -------
        data : dict
            Dictionary with same structure as generate_linear_scenario
        """
        # Generate moderators
        X = np.random.randn(n_studies, n_features)

        # Complex relationship:
        # X0: Quadratic effect
        # X1: Linear effect
        # X0 * X1: Interaction
        # X2: Threshold effect
        # X3: Weak linear effect
        # X4-X7: Noise variables (no true effect)

        theta_true = (
            0.4 +  # Intercept
            0.5 * X[:, 0] - 0.3 * X[:, 0]**2 +  # Quadratic effect
            0.4 * X[:, 1] +  # Linear effect
            0.3 * X[:, 0] * X[:, 1] +  # Interaction
            np.where(X[:, 2] > 0, 0.3, -0.1) +  # Threshold effect
            0.15 * X[:, 3]  # Weak linear effect
            # X4-X7 are noise (coefficient = 0)
        )

        # Add between-study heterogeneity
        tau = np.sqrt(tau2)
        theta = theta_true + np.random.randn(n_studies) * tau

        # Generate sample sizes and observed effects
        n_per_study = np.maximum(20, np.random.normal(mean_n, sd_n, n_studies).astype(int))
        se = np.sqrt(4 / n_per_study)
        y = theta + np.random.randn(n_studies) * se

        study_names = [f"Study_{i+1}" for i in range(n_studies)]

        # True variable importance (for evaluation)
        true_importance = np.array([0.8, 0.7, 0.4, 0.15, 0.0, 0.0, 0.0, 0.0])

        return {
            'X': X,
            'y': y,
            'se': se,
            'theta_true': theta_true,
            'tau2': tau2,
            'study_names': study_names,
            'scenario': 'complex',
            'true_importance': true_importance
        }


class SimulationStudy:
    """
    Conduct comprehensive simulation studies comparing BART with traditional methods.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize simulation study.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.simulator = MetaAnalysisSimulator(random_state=random_state)
        self.results = []

    def compare_methods_single_scenario(
        self,
        scenario_data: Dict[str, np.ndarray],
        methods: List[str] = ['bart', 'linear']
    ) -> Dict[str, float]:
        """
        Compare methods on a single simulated dataset.

        Parameters
        ----------
        scenario_data : dict
            Data generated by MetaAnalysisSimulator
        methods : list of str, default=['bart', 'linear']
            Methods to compare

        Returns
        -------
        metrics : dict
            Performance metrics for each method
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, r2_score

        X = scenario_data['X']
        y = scenario_data['y']
        se = scenario_data['se']
        theta_true = scenario_data['theta_true']

        weights = 1.0 / se**2
        metrics = {}

        # BART Meta-Regression
        if 'bart' in methods:
            try:
                from bart_meta_regression import BARTMetaRegression

                bart = BARTMetaRegression(
                    n_draws=1000,
                    n_tune=500,
                    random_state=self.random_state
                )
                bart.fit(X, y, se)
                bart_pred = bart.predictions

                metrics['BART_RMSE'] = np.sqrt(mean_squared_error(theta_true, bart_pred))
                metrics['BART_R2'] = r2_score(theta_true, bart_pred)
                metrics['BART_Coverage'] = self._calculate_coverage(
                    theta_true, bart_pred, se
                )
            except Exception as e:
                print(f"BART failed: {e}")
                metrics['BART_RMSE'] = np.nan
                metrics['BART_R2'] = np.nan
                metrics['BART_Coverage'] = np.nan

        # Linear Meta-Regression (weighted ridge)
        if 'linear' in methods:
            linear = Ridge(alpha=1.0)
            linear.fit(X, y, sample_weight=weights)
            linear_pred = linear.predict(X)

            metrics['Linear_RMSE'] = np.sqrt(mean_squared_error(theta_true, linear_pred))
            metrics['Linear_R2'] = r2_score(theta_true, linear_pred)
            metrics['Linear_Coverage'] = self._calculate_coverage(
                theta_true, linear_pred, se
            )

        return metrics

    def _calculate_coverage(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        se: np.ndarray,
        alpha: float = 0.05
    ) -> float:
        """Calculate coverage probability of confidence intervals."""
        z = stats.norm.ppf(1 - alpha / 2)
        lower = predictions - z * se
        upper = predictions + z * se
        coverage = np.mean((true_values >= lower) & (true_values <= upper))
        return coverage

    def run_monte_carlo_study(
        self,
        scenario_func: Callable,
        n_simulations: int = 100,
        **scenario_kwargs
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation study.

        Parameters
        ----------
        scenario_func : callable
            Function to generate scenario data
        n_simulations : int, default=100
            Number of simulation iterations
        **scenario_kwargs
            Additional arguments for scenario_func

        Returns
        -------
        results_df : DataFrame
            Aggregated simulation results
        """
        all_metrics = []

        for i in range(n_simulations):
            # Generate data
            data = scenario_func(**scenario_kwargs)

            # Compare methods
            metrics = self.compare_methods_single_scenario(data)
            metrics['iteration'] = i
            all_metrics.append(metrics)

        results_df = pd.DataFrame(all_metrics)
        return results_df

    def summarize_simulation_results(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Summarize simulation results across iterations.

        Parameters
        ----------
        results_df : DataFrame
            Results from run_monte_carlo_study

        Returns
        -------
        summary_df : DataFrame
            Summary statistics
        """
        summary_stats = []

        for col in results_df.columns:
            if col != 'iteration':
                summary_stats.append({
                    'Metric': col,
                    'Mean': results_df[col].mean(),
                    'SD': results_df[col].std(),
                    'Median': results_df[col].median(),
                    'Q25': results_df[col].quantile(0.25),
                    'Q75': results_df[col].quantile(0.75)
                })

        return pd.DataFrame(summary_stats)

    def plot_simulation_results(
        self,
        results_df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Visualize simulation results.

        Parameters
        ----------
        results_df : DataFrame
            Results from run_monte_carlo_study
        figsize : tuple, default=(14, 6)
            Figure size

        Returns
        -------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # RMSE comparison
        bart_rmse = results_df['BART_RMSE'].dropna()
        linear_rmse = results_df['Linear_RMSE'].dropna()

        axes[0].violinplot(
            [bart_rmse, linear_rmse],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )
        axes[0].set_xticks([1, 2])
        axes[0].set_xticklabels(['BART', 'Linear'])
        axes[0].set_ylabel('RMSE', fontweight='bold')
        axes[0].set_title('Prediction Error', fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')

        # R² comparison
        bart_r2 = results_df['BART_R2'].dropna()
        linear_r2 = results_df['Linear_R2'].dropna()

        axes[1].violinplot(
            [bart_r2, linear_r2],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(['BART', 'Linear'])
        axes[1].set_ylabel('R²', fontweight='bold')
        axes[1].set_title('Variance Explained', fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')

        # Coverage comparison
        bart_cov = results_df['BART_Coverage'].dropna()
        linear_cov = results_df['Linear_Coverage'].dropna()

        axes[2].violinplot(
            [bart_cov, linear_cov],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )
        axes[2].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Nominal')
        axes[2].set_xticks([1, 2])
        axes[2].set_xticklabels(['BART', 'Linear'])
        axes[2].set_ylabel('Coverage Probability', fontweight='bold')
        axes[2].set_title('95% CI Coverage', fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis='y')

        fig.suptitle('Simulation Study Results', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig


def run_comprehensive_simulation_study(
    n_simulations: int = 50,
    random_state: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive simulation study across multiple scenarios.

    Parameters
    ----------
    n_simulations : int, default=50
        Number of Monte Carlo iterations per scenario
    random_state : int, optional
        Random seed

    Returns
    -------
    all_results : dict
        Dictionary mapping scenario names to result DataFrames
    """
    simulator = MetaAnalysisSimulator(random_state=random_state)
    study = SimulationStudy(random_state=random_state)

    scenarios = {
        'Linear': lambda: simulator.generate_linear_scenario(n_studies=60),
        'Quadratic': lambda: simulator.generate_nonlinear_scenario(
            n_studies=60, nonlinear_type='quadratic'
        ),
        'Interaction': lambda: simulator.generate_interaction_scenario(n_studies=60),
        'Complex': lambda: simulator.generate_complex_scenario(n_studies=100)
    }

    all_results = {}

    for scenario_name, scenario_func in scenarios.items():
        print(f"\nRunning simulation: {scenario_name}")
        results = study.run_monte_carlo_study(
            scenario_func,
            n_simulations=n_simulations
        )
        all_results[scenario_name] = results

        # Print summary
        summary = study.summarize_simulation_results(results)
        print(f"\n{scenario_name} Scenario Summary:")
        print(summary.to_string(index=False))

    return all_results
