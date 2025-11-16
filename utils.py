"""
Utility Functions for BART Meta-Regression

Helper functions for data preparation, validation, and common meta-analytic
operations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Union
import warnings


def calculate_effect_size_variance(
    n1: np.ndarray,
    n2: np.ndarray,
    effect_type: str = 'SMD'
) -> np.ndarray:
    """
    Calculate variance of effect sizes.

    Parameters
    ----------
    n1, n2 : ndarray
        Sample sizes for groups 1 and 2
    effect_type : str, default='SMD'
        Type of effect size: 'SMD' (standardized mean difference),
        'OR' (odds ratio), 'RR' (risk ratio)

    Returns
    -------
    variance : ndarray
        Variance of effect sizes
    """
    if effect_type == 'SMD':
        # Hedges' g variance approximation
        return (n1 + n2) / (n1 * n2) + 0.5 / (n1 + n2)
    elif effect_type == 'OR':
        # Log odds ratio variance (requires event counts)
        warnings.warn("For OR, provide event counts directly")
        return np.nan
    elif effect_type == 'RR':
        warnings.warn("For RR, provide event counts directly")
        return np.nan
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")


def convert_or_to_d(log_or: np.ndarray) -> np.ndarray:
    """
    Convert log odds ratio to standardized mean difference (Cohen's d).

    Uses the approximation: d ≈ log(OR) * √3 / π

    Parameters
    ----------
    log_or : ndarray
        Log odds ratios

    Returns
    -------
    d : ndarray
        Approximate Cohen's d values
    """
    return log_or * np.sqrt(3) / np.pi


def convert_d_to_or(d: np.ndarray) -> np.ndarray:
    """
    Convert standardized mean difference to log odds ratio.

    Uses the approximation: log(OR) ≈ d * π / √3

    Parameters
    ----------
    d : ndarray
        Cohen's d values

    Returns
    -------
    log_or : ndarray
        Approximate log odds ratios
    """
    return d * np.pi / np.sqrt(3)


def hedges_g_correction(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Apply Hedges' g correction to Cohen's d for small sample bias.

    Parameters
    ----------
    d : ndarray
        Cohen's d values
    n : ndarray
        Total sample sizes (n1 + n2)

    Returns
    -------
    g : ndarray
        Hedges' g (bias-corrected d)
    """
    correction_factor = 1 - (3 / (4 * n - 9))
    return d * correction_factor


def cochrans_q(
    effect_sizes: np.ndarray,
    variances: np.ndarray
) -> Tuple[float, int, float]:
    """
    Calculate Cochran's Q test for heterogeneity.

    Parameters
    ----------
    effect_sizes : ndarray
        Observed effect sizes
    variances : ndarray
        Variances of effect sizes

    Returns
    -------
    Q : float
        Cochran's Q statistic
    df : int
        Degrees of freedom
    p_value : float
        P-value for Q test
    """
    weights = 1.0 / variances
    weighted_mean = np.sum(weights * effect_sizes) / np.sum(weights)
    Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
    df = len(effect_sizes) - 1
    p_value = 1 - stats.chi2.cdf(Q, df)

    return Q, df, p_value


def i_squared(Q: float, df: int) -> float:
    """
    Calculate I² statistic from Cochran's Q.

    Parameters
    ----------
    Q : float
        Cochran's Q statistic
    df : int
        Degrees of freedom

    Returns
    -------
    I2 : float
        I² statistic (percentage)
    """
    if Q <= df:
        return 0.0
    return 100 * (Q - df) / Q


def tau_squared_dl(
    effect_sizes: np.ndarray,
    variances: np.ndarray
) -> float:
    """
    Estimate between-study variance using DerSimonian-Laird method.

    Parameters
    ----------
    effect_sizes : ndarray
        Observed effect sizes
    variances : ndarray
        Variances of effect sizes

    Returns
    -------
    tau2 : float
        Estimated between-study variance
    """
    weights = 1.0 / variances
    weighted_mean = np.sum(weights * effect_sizes) / np.sum(weights)
    Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
    df = len(effect_sizes) - 1

    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)

    if C <= 0:
        return 0.0

    tau2 = max(0, (Q - df) / C)
    return tau2


def prediction_interval(
    effect_mean: float,
    tau2: float,
    se: float,
    alpha: float = 0.05,
    df: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate prediction interval for future studies.

    Parameters
    ----------
    effect_mean : float
        Pooled effect estimate
    tau2 : float
        Between-study variance
    se : float
        Standard error of pooled effect
    alpha : float, default=0.05
        Significance level
    df : int, optional
        Degrees of freedom (if None, use normal approximation)

    Returns
    -------
    lower, upper : float
        Prediction interval bounds
    """
    pred_var = se**2 + tau2

    if df is not None:
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        lower = effect_mean - t_crit * np.sqrt(pred_var)
        upper = effect_mean + t_crit * np.sqrt(pred_var)
    else:
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lower = effect_mean - z_crit * np.sqrt(pred_var)
        upper = effect_mean + z_crit * np.sqrt(pred_var)

    return lower, upper


def validate_meta_data(
    effect_sizes: np.ndarray,
    variances: Optional[np.ndarray] = None,
    standard_errors: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize meta-analytic data.

    Parameters
    ----------
    effect_sizes : ndarray
        Effect sizes
    variances : ndarray, optional
        Variances of effect sizes
    standard_errors : ndarray, optional
        Standard errors of effect sizes

    Returns
    -------
    effect_sizes : ndarray
        Validated effect sizes
    standard_errors : ndarray
        Standard errors

    Raises
    ------
    ValueError
        If data validation fails
    """
    effect_sizes = np.asarray(effect_sizes, dtype=float)

    if variances is not None and standard_errors is not None:
        raise ValueError("Provide either variances or standard_errors, not both")

    if variances is not None:
        variances = np.asarray(variances, dtype=float)
        if np.any(variances <= 0):
            raise ValueError("All variances must be positive")
        standard_errors = np.sqrt(variances)
    elif standard_errors is not None:
        standard_errors = np.asarray(standard_errors, dtype=float)
        if np.any(standard_errors <= 0):
            raise ValueError("All standard errors must be positive")
    else:
        raise ValueError("Must provide either variances or standard_errors")

    if len(effect_sizes) != len(standard_errors):
        raise ValueError("effect_sizes and standard_errors must have same length")

    if np.any(np.isnan(effect_sizes)) or np.any(np.isnan(standard_errors)):
        raise ValueError("Data contains NaN values")

    if np.any(np.isinf(effect_sizes)) or np.any(np.isinf(standard_errors)):
        raise ValueError("Data contains infinite values")

    return effect_sizes, standard_errors


def trim_and_fill(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    side: str = 'auto'
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply trim-and-fill method for publication bias adjustment.

    This is a simplified implementation. For production use,
    consider using established packages like metafor (R) or punimeta.

    Parameters
    ----------
    effect_sizes : ndarray
        Observed effect sizes
    variances : ndarray
        Variances of effect sizes
    side : str, default='auto'
        Side to impute: 'left', 'right', or 'auto'

    Returns
    -------
    adjusted_effects : ndarray
        Effect sizes with imputed studies
    adjusted_variances : ndarray
        Variances with imputed studies
    n_imputed : int
        Number of imputed studies

    References
    ----------
    Duval, S., & Tweedie, R. (2000). Trim and fill: a simple funnel‐plot–based
    method of testing and adjusting for publication bias in meta‐analysis.
    Biometrics, 56(2), 455-463.
    """
    # Rank studies by effect size
    order = np.argsort(effect_sizes)
    sorted_effects = effect_sizes[order]
    sorted_vars = variances[order]

    # Estimate number of missing studies (simplified)
    # In practice, use iterative algorithm
    n = len(effect_sizes)
    center_idx = n // 2

    # Count asymmetry
    left_count = np.sum(sorted_effects < sorted_effects[center_idx])
    right_count = np.sum(sorted_effects > sorted_effects[center_idx])

    if side == 'auto':
        side = 'left' if left_count < right_count else 'right'

    n_imputed = abs(left_count - right_count) // 2

    if n_imputed == 0:
        return effect_sizes, variances, 0

    # Impute missing studies by reflecting
    if side == 'left':
        # Reflect smallest effects to left side
        imputed_effects = 2 * sorted_effects[center_idx] - sorted_effects[-n_imputed:][::-1]
        imputed_vars = sorted_vars[-n_imputed:][::-1]
    else:
        # Reflect largest effects to right side
        imputed_effects = 2 * sorted_effects[center_idx] - sorted_effects[:n_imputed][::-1]
        imputed_vars = sorted_vars[:n_imputed][::-1]

    adjusted_effects = np.concatenate([effect_sizes, imputed_effects])
    adjusted_variances = np.concatenate([variances, imputed_vars])

    return adjusted_effects, adjusted_variances, n_imputed


def eggers_test(
    effect_sizes: np.ndarray,
    standard_errors: np.ndarray
) -> Tuple[float, float, float]:
    """
    Perform Egger's test for publication bias.

    Tests whether funnel plot asymmetry is significant.

    Parameters
    ----------
    effect_sizes : ndarray
        Observed effect sizes
    standard_errors : ndarray
        Standard errors

    Returns
    -------
    intercept : float
        Regression intercept (bias measure)
    t_statistic : float
        t-statistic for intercept
    p_value : float
        P-value for intercept test

    References
    ----------
    Egger, M., et al. (1997). Bias in meta-analysis detected by a simple,
    graphical test. BMJ, 315(7109), 629-634.
    """
    precision = 1.0 / standard_errors
    standardized_effects = effect_sizes * precision

    # Regression: standardized_effects ~ precision
    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(precision, standardized_effects)

    # t-test for intercept
    t_statistic = intercept / std_err

    return intercept, t_statistic, p_value


def failsafe_n(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    alpha: float = 0.05
) -> int:
    """
    Calculate Rosenthal's fail-safe N.

    Estimates number of null studies needed to make result non-significant.

    Parameters
    ----------
    effect_sizes : ndarray
        Observed effect sizes
    variances : ndarray
        Variances of effect sizes
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    failsafe_n : int
        Number of null studies needed

    References
    ----------
    Rosenthal, R. (1979). The file drawer problem and tolerance for null results.
    Psychological Bulletin, 86(3), 638.
    """
    # Calculate overall Z-score
    weights = 1.0 / variances
    weighted_mean = np.sum(weights * effect_sizes) / np.sum(weights)
    se = 1.0 / np.sqrt(np.sum(weights))

    z_observed = weighted_mean / se
    z_critical = stats.norm.ppf(1 - alpha / 2)

    if z_observed <= z_critical:
        return 0  # Already non-significant

    # Calculate fail-safe N
    k = len(effect_sizes)
    failsafe = int(k * (z_observed**2 / z_critical**2 - 1))

    return max(0, failsafe)


def check_influential_studies(
    effect_sizes: np.ndarray,
    standard_errors: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Identify potentially influential studies using standardized residuals.

    Parameters
    ----------
    effect_sizes : ndarray
        Observed effect sizes
    standard_errors : ndarray
        Standard errors
    threshold : float, default=3.0
        Standardized residual threshold

    Returns
    -------
    influential : ndarray
        Boolean array indicating influential studies
    """
    weights = 1.0 / standard_errors**2
    weighted_mean = np.average(effect_sizes, weights=weights)

    residuals = effect_sizes - weighted_mean
    standardized_residuals = residuals / standard_errors

    influential = np.abs(standardized_residuals) > threshold

    return influential


def power_analysis_meta_regression(
    n_studies: int,
    effect_size: float,
    tau2: float,
    mean_n_per_study: int,
    alpha: float = 0.05
) -> float:
    """
    Estimate power for detecting a meta-regression effect.

    Simplified power calculation for meta-regression.

    Parameters
    ----------
    n_studies : int
        Number of studies
    effect_size : float
        Expected regression coefficient
    tau2 : float
        Expected between-study heterogeneity
    mean_n_per_study : int
        Average sample size per study
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    power : float
        Estimated statistical power
    """
    # Approximate variance of regression coefficient
    var_per_study = 4.0 / mean_n_per_study  # Approximate for SMD
    total_var = var_per_study + tau2

    se_beta = np.sqrt(total_var / n_studies)

    # Non-centrality parameter
    ncp = effect_size / se_beta

    # Critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Power
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    return power


def format_ci(
    estimate: float,
    lower: float,
    upper: float,
    decimals: int = 3
) -> str:
    """
    Format estimate with confidence interval for publication.

    Parameters
    ----------
    estimate : float
        Point estimate
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    decimals : int, default=3
        Number of decimal places

    Returns
    -------
    formatted : str
        Formatted string (e.g., "0.45 [0.32, 0.58]")
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(estimate)} [{fmt.format(lower)}, {fmt.format(upper)}]"
