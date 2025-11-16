"""
Comprehensive Example: BART Meta-Regression Analysis

This script demonstrates the complete workflow for publication-quality
BART meta-regression analysis including:

1. Data generation and preparation
2. Model fitting with BART
3. Diagnostic analysis
4. Variable importance assessment
5. Partial dependence visualization
6. Heterogeneity assessment
7. Model comparison
8. Publication-quality figures

Run this script to see BART meta-regression in action!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from bart_meta_regression import BARTMetaRegression, compare_with_linear_meta_regression
from visualization import MetaRegressionVisualizer
from simulation_studies import MetaAnalysisSimulator, SimulationStudy

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def example_1_linear_meta_regression():
    """
    Example 1: Simple linear meta-regression

    This example demonstrates BART on a simple linear scenario
    to establish baseline functionality.
    """
    print("=" * 80)
    print("EXAMPLE 1: Linear Meta-Regression")
    print("=" * 80)

    # Generate simulated data
    simulator = MetaAnalysisSimulator(random_state=RANDOM_STATE)
    data = simulator.generate_linear_scenario(
        n_studies=40,
        n_features=3,
        tau2=0.04
    )

    # Create feature names
    feature_names = ['Age (years)', 'Study Quality', 'Publication Year']

    # Fit BART meta-regression
    print("\nFitting BART meta-regression model...")
    bart = BARTMetaRegression(
        n_trees=50,
        n_draws=2000,
        n_tune=1000,
        random_state=RANDOM_STATE
    )

    bart.fit(
        X=data['X'],
        y=data['y'],
        se=data['se'],
        feature_names=feature_names
    )

    # Print model summary
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    summary = bart.summary()
    print(summary.to_string(index=False))

    # Variable importance
    print("\n" + "=" * 60)
    print("Variable Importance")
    print("=" * 60)
    importance = bart.variable_importance(method='permutation', n_repeats=5)
    print(importance.to_string(index=False))

    # Heterogeneity statistics
    print("\n" + "=" * 60)
    print("Heterogeneity Assessment")
    print("=" * 60)
    het_stats = bart.heterogeneity_stats()
    for key, value in het_stats.items():
        print(f"{key:12s}: {value:.4f}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Forest plot
    viz = MetaRegressionVisualizer()
    fig1 = viz.forest_plot(
        study_names=data['study_names'],
        effect_sizes=data['y'],
        standard_errors=data['se'],
        predictions=bart.predictions,
        figsize=(12, 8)
    )
    plt.savefig('example1_forest_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Forest plot saved: example1_forest_plot.png")
    plt.close()

    # Diagnostic panel
    fig2 = viz.diagnostic_panel(bart, figsize=(16, 12))
    plt.savefig('example1_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ Diagnostic panel saved: example1_diagnostics.png")
    plt.close()

    # Variable importance plot
    fig3 = bart.plot_variable_importance(method='permutation', figsize=(10, 6))
    plt.savefig('example1_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Variable importance saved: example1_importance.png")
    plt.close()

    print("\n✓ Example 1 completed successfully!\n")
    return bart, data


def example_2_nonlinear_meta_regression():
    """
    Example 2: Non-linear meta-regression

    This example demonstrates BART's ability to detect non-linear
    relationships without pre-specification.
    """
    print("=" * 80)
    print("EXAMPLE 2: Non-Linear Meta-Regression (Quadratic Effect)")
    print("=" * 80)

    # Generate data with quadratic relationship
    simulator = MetaAnalysisSimulator(random_state=RANDOM_STATE)
    data = simulator.generate_nonlinear_scenario(
        n_studies=60,
        n_features=4,
        tau2=0.06,
        nonlinear_type='quadratic'
    )

    feature_names = ['Dose (mg)', 'Duration (weeks)', 'Age (years)', 'Baseline Severity']

    # Fit BART model
    print("\nFitting BART meta-regression model...")
    bart = BARTMetaRegression(
        n_trees=75,
        n_draws=2000,
        n_tune=1000,
        random_state=RANDOM_STATE
    )

    bart.fit(
        X=data['X'],
        y=data['y'],
        se=data['se'],
        feature_names=feature_names
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(bart.summary().to_string(index=False))

    # Partial dependence plots (especially for the non-linear feature)
    print("\nGenerating partial dependence plots...")
    viz = MetaRegressionVisualizer()

    fig = viz.partial_dependence_grid(
        bart,
        feature_indices=[0, 1, 2, 3],  # All features
        figsize=(14, 10)
    )
    plt.savefig('example2_partial_dependence.png', dpi=300, bbox_inches='tight')
    print("✓ Partial dependence plots saved: example2_partial_dependence.png")
    plt.close()

    # Focus on the non-linear feature
    fig = bart.plot_partial_dependence(0, figsize=(10, 7))
    plt.savefig('example2_dose_response.png', dpi=300, bbox_inches='tight')
    print("✓ Dose-response curve saved: example2_dose_response.png")
    plt.close()

    # Funnel plot
    fig = viz.funnel_plot(
        effect_sizes=data['y'],
        standard_errors=data['se'],
        predictions=bart.predictions,
        figsize=(10, 8)
    )
    plt.savefig('example2_funnel_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Funnel plot saved: example2_funnel_plot.png")
    plt.close()

    print("\n✓ Example 2 completed successfully!\n")
    return bart, data


def example_3_interaction_detection():
    """
    Example 3: Interaction detection

    This example shows how BART automatically detects interactions
    without requiring pre-specification.
    """
    print("=" * 80)
    print("EXAMPLE 3: Interaction Detection")
    print("=" * 80)

    # Generate data with interaction
    simulator = MetaAnalysisSimulator(random_state=RANDOM_STATE)
    data = simulator.generate_interaction_scenario(
        n_studies=70,
        n_features=4,
        tau2=0.05,
        interaction_strength=0.6
    )

    feature_names = ['Treatment Intensity', 'Patient Adherence', 'Age', 'Comorbidities']

    # Fit BART model
    print("\nFitting BART meta-regression model...")
    bart = BARTMetaRegression(
        n_trees=75,
        n_draws=2000,
        n_tune=1000,
        random_state=RANDOM_STATE
    )

    bart.fit(
        X=data['X'],
        y=data['y'],
        se=data['se'],
        feature_names=feature_names
    )

    # Variable importance
    print("\n" + "=" * 60)
    print("Variable Importance")
    print("=" * 60)
    importance = bart.variable_importance(method='permutation', n_repeats=10)
    print(importance.to_string(index=False))

    # Interaction strength analysis
    print("\n" + "=" * 60)
    print("Pairwise Interaction Strengths")
    print("=" * 60)

    interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            strength = bart.interaction_strength(i, j, n_samples=100)
            interactions.append({
                'Feature 1': feature_names[i],
                'Feature 2': feature_names[j],
                'Interaction Strength': strength
            })

    interaction_df = pd.DataFrame(interactions)
    interaction_df = interaction_df.sort_values('Interaction Strength', ascending=False)
    print(interaction_df.to_string(index=False))

    # Visualizations
    print("\nGenerating visualizations...")
    viz = MetaRegressionVisualizer()

    # Interaction heatmap
    fig = viz.interaction_heatmap(
        X=data['X'],
        feature_names=feature_names,
        figsize=(10, 8)
    )
    plt.savefig('example3_interaction_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Interaction heatmap saved: example3_interaction_heatmap.png")
    plt.close()

    # Partial dependence for interacting variables
    fig = viz.partial_dependence_grid(
        bart,
        feature_indices=[0, 1],  # The two interacting features
        figsize=(12, 5)
    )
    plt.savefig('example3_interacting_features.png', dpi=300, bbox_inches='tight')
    print("✓ Partial dependence (interacting features) saved")
    plt.close()

    print("\n✓ Example 3 completed successfully!\n")
    return bart, data


def example_4_model_comparison():
    """
    Example 4: BART vs Linear Meta-Regression

    This example compares BART with traditional linear meta-regression
    using cross-validation.
    """
    print("=" * 80)
    print("EXAMPLE 4: Model Comparison (BART vs Linear Meta-Regression)")
    print("=" * 80)

    # Generate complex scenario
    simulator = MetaAnalysisSimulator(random_state=RANDOM_STATE)
    data = simulator.generate_complex_scenario(
        n_studies=80,
        n_features=6,
        tau2=0.07
    )

    print("\nComparing BART and Linear meta-regression using 5-fold CV...")
    print("(This may take a few minutes...)")

    # Compare methods
    comparison = compare_with_linear_meta_regression(
        X=data['X'],
        y=data['y'],
        se=data['se'],
        cv_folds=5,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print(comparison.to_string(index=False))

    # Visualize comparison
    viz = MetaRegressionVisualizer()
    fig = viz.comparative_performance_plot(comparison, figsize=(12, 6))
    plt.savefig('example4_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Model comparison plot saved: example4_model_comparison.png")
    plt.close()

    print("\n✓ Example 4 completed successfully!\n")
    return comparison


def example_5_simulation_study():
    """
    Example 5: Monte Carlo Simulation Study

    This example runs a comprehensive simulation study to evaluate
    BART's performance across different scenarios.
    """
    print("=" * 80)
    print("EXAMPLE 5: Monte Carlo Simulation Study")
    print("=" * 80)
    print("\nRunning 20 simulations across 4 scenarios...")
    print("(This will take several minutes...)\n")

    # Run simulation study
    from simulation_studies import run_comprehensive_simulation_study

    results = run_comprehensive_simulation_study(
        n_simulations=20,  # Reduced for faster demonstration
        random_state=RANDOM_STATE
    )

    # Visualize results for each scenario
    study = SimulationStudy(random_state=RANDOM_STATE)

    for scenario_name, scenario_results in results.items():
        fig = study.plot_simulation_results(scenario_results, figsize=(15, 6))
        filename = f'example5_simulation_{scenario_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Simulation results saved: {filename}")
        plt.close()

    print("\n✓ Example 5 completed successfully!\n")
    return results


def run_all_examples():
    """Run all examples sequentially."""
    print("\n" + "=" * 80)
    print("BART META-REGRESSION: COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("\nThis script will run 5 comprehensive examples demonstrating")
    print("BART meta-regression for publication-quality analysis.\n")

    # Example 1: Linear
    bart1, data1 = example_1_linear_meta_regression()

    # Example 2: Non-linear
    bart2, data2 = example_2_nonlinear_meta_regression()

    # Example 3: Interactions
    bart3, data3 = example_3_interaction_detection()

    # Example 4: Model comparison
    comparison = example_4_model_comparison()

    # Example 5: Simulation study (optional - can be commented out for speed)
    # results = example_5_simulation_study()

    print("=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - example1_*.png: Linear meta-regression results")
    print("  - example2_*.png: Non-linear meta-regression results")
    print("  - example3_*.png: Interaction detection results")
    print("  - example4_*.png: Model comparison results")
    print("  - example5_*.png: Simulation study results (if run)")
    print("\nThese figures are publication-ready and can be used directly")
    print("in manuscripts or presentations.")
    print("=" * 80)


if __name__ == "__main__":
    # Run all examples
    run_all_examples()

    # Or run individual examples:
    # example_1_linear_meta_regression()
    # example_2_nonlinear_meta_regression()
    # example_3_interaction_detection()
    # example_4_model_comparison()
    # example_5_simulation_study()
