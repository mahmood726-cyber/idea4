"""
Advanced Visualization Tools for BART Meta-Regression

Publication-quality visualizations for meta-analytic results including:
- Enhanced forest plots with BART predictions
- Interaction heatmaps
- Funnel plots with heterogeneity assessment
- Multi-panel diagnostic plots
- Variable importance plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, List, Tuple, Union
import warnings

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class MetaRegressionVisualizer:
    """Advanced visualization tools for BART meta-regression."""

    @staticmethod
    def forest_plot(
        study_names: List[str],
        effect_sizes: np.ndarray,
        standard_errors: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        pred_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        figsize: Tuple[int, int] = (12, 10),
        sort_by: str = 'effect_size'
    ) -> plt.Figure:
        """
        Create enhanced forest plot with BART predictions.

        Parameters
        ----------
        study_names : list of str
            Study identifiers
        effect_sizes : ndarray
            Observed effect sizes
        standard_errors : ndarray
            Standard errors
        predictions : ndarray, optional
            BART predicted effect sizes
        pred_intervals : tuple of ndarray, optional
            (lower, upper) credible intervals for predictions
        figsize : tuple, default=(12, 10)
            Figure size
        sort_by : str, default='effect_size'
            Sort studies by 'effect_size', 'prediction', or 'none'

        Returns
        -------
        fig : matplotlib Figure
        """
        n_studies = len(study_names)

        # Calculate confidence intervals
        ci_lower = effect_sizes - 1.96 * standard_errors
        ci_upper = effect_sizes + 1.96 * standard_errors

        # Create DataFrame for sorting
        df = pd.DataFrame({
            'study': study_names,
            'effect': effect_sizes,
            'se': standard_errors,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

        if predictions is not None:
            df['prediction'] = predictions

        # Sort studies
        if sort_by == 'effect_size':
            df = df.sort_values('effect')
        elif sort_by == 'prediction' and predictions is not None:
            df = df.sort_values('prediction')

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(n_studies)

        # Plot observed effects with confidence intervals
        colors_obs = plt.cm.Blues(0.6)
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.plot(
                [row['ci_lower'], row['ci_upper']],
                [i, i],
                'o-',
                color=colors_obs,
                linewidth=2,
                markersize=8,
                alpha=0.7,
                label='Observed (95% CI)' if i == 0 else ''
            )

        # Plot BART predictions
        if predictions is not None:
            colors_pred = plt.cm.Reds(0.6)
            for i, (idx, row) in enumerate(df.iterrows()):
                ax.plot(
                    row['prediction'],
                    i,
                    'D',
                    color=colors_pred,
                    markersize=10,
                    alpha=0.8,
                    label='BART Prediction' if i == 0 else ''
                )

                # Add prediction intervals if available
                if pred_intervals is not None:
                    ax.plot(
                        [pred_intervals[0][i], pred_intervals[1][i]],
                        [i, i],
                        '-',
                        color=colors_pred,
                        linewidth=1.5,
                        alpha=0.4
                    )

        # Add null effect line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['study'])
        ax.set_xlabel('Effect Size', fontsize=13, fontweight='bold')
        ax.set_ylabel('Study', fontsize=13, fontweight='bold')
        ax.set_title('Forest Plot: Observed Effects vs BART Predictions',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(axis='x', alpha=0.3, linestyle=':')

        plt.tight_layout()
        return fig

    @staticmethod
    def funnel_plot(
        effect_sizes: np.ndarray,
        standard_errors: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create funnel plot with contours and BART predictions.

        Parameters
        ----------
        effect_sizes : ndarray
            Observed effect sizes
        standard_errors : ndarray
            Standard errors
        predictions : ndarray, optional
            BART predicted effect sizes
        figsize : tuple, default=(10, 8)
            Figure size

        Returns
        -------
        fig : matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate pooled effect (weighted mean)
        weights = 1.0 / standard_errors**2
        pooled_effect = np.average(effect_sizes, weights=weights)

        # Plot studies
        ax.scatter(
            effect_sizes,
            standard_errors,
            s=100,
            alpha=0.6,
            c='steelblue',
            edgecolors='black',
            linewidth=1,
            label='Observed Studies'
        )

        # Plot BART predictions
        if predictions is not None:
            ax.scatter(
                predictions,
                standard_errors,
                s=120,
                alpha=0.7,
                c='coral',
                marker='D',
                edgecolors='darkred',
                linewidth=1,
                label='BART Predictions'
            )

        # Add funnel contours (95% CI)
        se_range = np.linspace(0, max(standard_errors) * 1.1, 100)
        lower_ci = pooled_effect - 1.96 * se_range
        upper_ci = pooled_effect + 1.96 * se_range

        ax.plot(lower_ci, se_range, '--', color='gray', linewidth=2, alpha=0.6)
        ax.plot(upper_ci, se_range, '--', color='gray', linewidth=2, alpha=0.6)
        ax.fill_betweenx(se_range, lower_ci, upper_ci, alpha=0.1, color='gray',
                         label='95% CI Contour')

        # Add pooled effect line
        ax.axvline(x=pooled_effect, color='red', linestyle='-',
                  linewidth=2, alpha=0.7, label=f'Pooled Effect = {pooled_effect:.3f}')

        # Invert y-axis (smaller SE at top)
        ax.invert_yaxis()

        # Customize plot
        ax.set_xlabel('Effect Size', fontsize=13, fontweight='bold')
        ax.set_ylabel('Standard Error', fontsize=13, fontweight='bold')
        ax.set_title('Funnel Plot: Publication Bias Assessment',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(alpha=0.3, linestyle=':')

        plt.tight_layout()
        return fig

    @staticmethod
    def interaction_heatmap(
        X: np.ndarray,
        feature_names: List[str],
        interaction_matrix: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create heatmap of feature interactions.

        Parameters
        ----------
        X : ndarray
            Feature matrix
        feature_names : list of str
            Feature names
        interaction_matrix : ndarray, optional
            Pre-computed interaction strengths. If None, computed from correlations.
        figsize : tuple, default=(10, 8)
            Figure size

        Returns
        -------
        fig : matplotlib Figure
        """
        if interaction_matrix is None:
            # Compute pairwise correlations as proxy for interactions
            interaction_matrix = np.abs(np.corrcoef(X.T))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            interaction_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Interaction Strength', 'shrink': 0.8},
            xticklabels=feature_names,
            yticklabels=feature_names,
            ax=ax
        )

        ax.set_title('Feature Interaction Heatmap',
                    fontsize=15, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    @staticmethod
    def diagnostic_panel(
        bart_model,
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Create comprehensive diagnostic panel.

        Parameters
        ----------
        bart_model : BARTMetaRegression
            Fitted BART model
        figsize : tuple, default=(16, 12)
            Figure size

        Returns
        -------
        fig : matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Residuals vs Fitted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(
            bart_model.predictions,
            bart_model.residuals,
            s=100 / bart_model.se_train,
            alpha=0.6,
            c='steelblue',
            edgecolors='black'
        )
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted', fontweight='bold')
        ax1.grid(alpha=0.3)

        # 2. Q-Q Plot
        ax2 = fig.add_subplot(gs[0, 1])
        stats.probplot(bart_model.residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot', fontweight='bold')
        ax2.grid(alpha=0.3)

        # 3. Scale-Location Plot
        ax3 = fig.add_subplot(gs[0, 2])
        standardized_residuals = bart_model.residuals / bart_model.se_train
        ax3.scatter(
            bart_model.predictions,
            np.sqrt(np.abs(standardized_residuals)),
            s=100 / bart_model.se_train,
            alpha=0.6,
            c='steelblue',
            edgecolors='black'
        )
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Standardized Residuals|')
        ax3.set_title('Scale-Location Plot', fontweight='bold')
        ax3.grid(alpha=0.3)

        # 4. Residual Histogram
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(bart_model.residuals, bins=20, alpha=0.7, color='steelblue',
                edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Residual Distribution', fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')

        # 5. Observed vs Predicted
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(
            bart_model.y_train,
            bart_model.predictions,
            s=100 / bart_model.se_train,
            alpha=0.6,
            c='steelblue',
            edgecolors='black'
        )
        # Add diagonal line
        lims = [
            np.min([ax5.get_xlim(), ax5.get_ylim()]),
            np.max([ax5.get_xlim(), ax5.get_ylim()]),
        ]
        ax5.plot(lims, lims, 'r--', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Observed Effect Size')
        ax5.set_ylabel('Predicted Effect Size')
        ax5.set_title('Observed vs Predicted', fontweight='bold')
        ax5.grid(alpha=0.3)

        # 6. Cook's Distance (simplified)
        ax6 = fig.add_subplot(gs[1, 2])
        leverage = 1.0 / bart_model.se_train**2
        leverage = leverage / leverage.sum()
        cooks_d = leverage * bart_model.residuals**2
        ax6.stem(range(len(cooks_d)), cooks_d, linefmt='steelblue',
                markerfmt='o', basefmt=' ')
        ax6.axhline(y=4/len(cooks_d), color='red', linestyle='--',
                   linewidth=2, label='Threshold')
        ax6.set_xlabel('Study Index')
        ax6.set_ylabel("Cook's Distance")
        ax6.set_title("Influence: Cook's Distance", fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3, axis='y')

        # 7. Variable Importance
        ax7 = fig.add_subplot(gs[2, :])
        importance_df = bart_model.variable_importance(method='permutation')
        ax7.barh(
            importance_df['feature'],
            importance_df['importance'],
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        ax7.set_xlabel('Importance Score')
        ax7.set_title('Variable Importance', fontweight='bold')
        ax7.invert_yaxis()
        ax7.grid(alpha=0.3, axis='x')

        fig.suptitle('BART Meta-Regression Diagnostic Panel',
                    fontsize=18, fontweight='bold', y=0.995)

        return fig

    @staticmethod
    def comparative_performance_plot(
        comparison_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot comparative performance of BART vs linear meta-regression.

        Parameters
        ----------
        comparison_df : DataFrame
            Comparison results from compare_with_linear_meta_regression
        figsize : tuple, default=(12, 6)
            Figure size

        Returns
        -------
        fig : matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        models = comparison_df['Model'].values
        x = np.arange(len(models))
        width = 0.35

        # Extract values and errors from strings
        def parse_metric(metric_str):
            mean, std = metric_str.split(' ± ')
            return float(mean), float(std)

        # RMSE comparison
        rmse_values = [parse_metric(v) for v in comparison_df['RMSE (mean ± std)']]
        rmse_means = [v[0] for v in rmse_values]
        rmse_stds = [v[1] for v in rmse_values]

        axes[0].bar(
            x,
            rmse_means,
            width,
            yerr=rmse_stds,
            alpha=0.7,
            capsize=5,
            color=['steelblue', 'coral'],
            edgecolor='black',
            linewidth=1.5
        )
        axes[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
        axes[0].set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=15, ha='right')
        axes[0].grid(alpha=0.3, axis='y')

        # R² comparison
        r2_values = [parse_metric(v) for v in comparison_df['R² (mean ± std)']]
        r2_means = [v[0] for v in r2_values]
        r2_stds = [v[1] for v in r2_values]

        axes[1].bar(
            x,
            r2_means,
            width,
            yerr=r2_stds,
            alpha=0.7,
            capsize=5,
            color=['steelblue', 'coral'],
            edgecolor='black',
            linewidth=1.5
        )
        axes[1].set_ylabel('R²', fontsize=12, fontweight='bold')
        axes[1].set_title('Coefficient of Determination', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=15, ha='right')
        axes[1].grid(alpha=0.3, axis='y')

        fig.suptitle('Model Performance Comparison',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        return fig

    @staticmethod
    def partial_dependence_grid(
        bart_model,
        feature_indices: List[Union[int, str]],
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create grid of partial dependence plots.

        Parameters
        ----------
        bart_model : BARTMetaRegression
            Fitted BART model
        feature_indices : list
            List of feature indices or names to plot
        figsize : tuple, optional
            Figure size. Auto-calculated if None.

        Returns
        -------
        fig : matplotlib Figure
        """
        n_features = len(feature_indices)
        n_cols = min(3, n_features)
        n_rows = int(np.ceil(n_features / n_cols))

        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feature_idx in enumerate(feature_indices):
            ax = axes[i]

            grid_values, pd_mean, pd_lower, pd_upper = \
                bart_model.partial_dependence(feature_idx)

            if isinstance(feature_idx, int):
                feature_name = bart_model.feature_names[feature_idx]
            else:
                feature_name = feature_idx
                feature_idx = bart_model.feature_names.index(feature_idx)

            # Plot partial dependence
            ax.plot(grid_values, pd_mean, 'b-', linewidth=2.5, label='Mean effect')
            ax.fill_between(
                grid_values,
                pd_lower,
                pd_upper,
                alpha=0.3,
                color='blue',
                label='95% CI'
            )

            # Add rug plot
            feature_values = bart_model.X_train[:, feature_idx]
            ax.plot(
                feature_values,
                np.ones_like(feature_values) * ax.get_ylim()[0],
                '|',
                color='gray',
                alpha=0.5,
                markersize=10
            )

            ax.set_xlabel(feature_name, fontsize=11, fontweight='bold')
            ax.set_ylabel('Partial Effect', fontsize=11, fontweight='bold')
            ax.set_title(f'PDP: {feature_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        fig.suptitle('Partial Dependence Plots',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig
