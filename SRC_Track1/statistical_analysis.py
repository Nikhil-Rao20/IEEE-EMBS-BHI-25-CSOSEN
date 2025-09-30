"""
🔬 Statistical Analysis Module
============================

Implements comprehensive statistical analysis for model comparison and validation.
Provides rigorous statistical testing for conference-quality results.

Features:
- Statistical significance testing
- Effect size calculations
- Confidence intervals
- Power analysis
- Multiple comparisons correction
- Clinical significance assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import statistical libraries
try:
    from scipy import stats
    from scipy.stats import (
        ttest_rel, wilcoxon, friedmanchisquare, ranksums,
        normaltest, levene, shapiro
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available. Install with: pip install scipy")

try:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.power import ttest_power
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available for advanced statistical tests.")

class StatisticalAnalysis:
    """
    Comprehensive statistical analysis for model comparison.
    
    Provides rigorous statistical testing suitable for academic publication.
    """
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        """
        Initialize statistical analysis.
        
        Args:
            alpha: Significance level for hypothesis tests
            confidence_level: Confidence level for intervals
        """
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.results = {}
        
        print(f"📊 Statistical Analysis Initialized")
        print(f"🎯 Significance level (α): {alpha}")
        print(f"📏 Confidence level: {confidence_level}")
    
    def test_normality(self, data: np.ndarray, test_name: str = "shapiro") -> Dict[str, Any]:
        """
        Test for normality in the data.
        
        Args:
            data: Data to test
            test_name: Type of normality test ('shapiro', 'normaltest')
            
        Returns:
            Dictionary with test results
        """
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available"}
        
        results = {}
        
        try:
            if test_name == "shapiro" and len(data) <= 5000:
                stat, p_value = shapiro(data)
                results['test'] = 'Shapiro-Wilk'
            else:
                stat, p_value = normaltest(data)
                results['test'] = 'D\'Agostino-Pearson'
            
            results['statistic'] = stat
            results['p_value'] = p_value
            results['is_normal'] = p_value > self.alpha
            results['interpretation'] = (
                "Data appears normally distributed" if results['is_normal']
                else "Data does not appear normally distributed"
            )
            
        except Exception as e:
            results['error'] = f"Normality test failed: {e}"
        
        return results
    
    def compare_two_models(self, scores1: np.ndarray, scores2: np.ndarray,
                          model1_name: str, model2_name: str,
                          metric_name: str = "MAE") -> Dict[str, Any]:
        """
        Compare two models using appropriate statistical tests.
        
        Args:
            scores1: Performance scores for model 1
            scores2: Performance scores for model 2
            model1_name: Name of first model
            model2_name: Name of second model
            metric_name: Name of the metric being compared
            
        Returns:
            Dictionary with comparison results
        """
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available"}
        
        results = {
            'model1': model1_name,
            'model2': model2_name,
            'metric': metric_name,
            'n_samples': len(scores1)
        }
        
        # Basic descriptive statistics
        results['model1_mean'] = np.mean(scores1)
        results['model1_std'] = np.std(scores1, ddof=1)
        results['model2_mean'] = np.mean(scores2)
        results['model2_std'] = np.std(scores2, ddof=1)
        results['difference'] = results['model1_mean'] - results['model2_mean']
        
        # Check if data is paired (same length)
        if len(scores1) != len(scores2):
            results['error'] = "Scores must be paired (same length)"
            return results
        
        # Test normality
        norm1 = self.test_normality(scores1)
        norm2 = self.test_normality(scores2)
        results['normality'] = {
            'model1_normal': norm1.get('is_normal', False),
            'model2_normal': norm2.get('is_normal', False)
        }
        
        both_normal = (norm1.get('is_normal', False) and 
                      norm2.get('is_normal', False))
        
        try:
            # Choose appropriate test
            if both_normal:
                # Paired t-test for normally distributed data
                stat, p_value = ttest_rel(scores1, scores2)
                results['test_used'] = 'Paired t-test'
                results['test_type'] = 'parametric'
            else:
                # Wilcoxon signed-rank test for non-normal data
                stat, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
                results['test_used'] = 'Wilcoxon signed-rank test'
                results['test_type'] = 'non-parametric'
            
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < self.alpha
            
            # Effect size calculation
            results['effect_size'] = self.calculate_effect_size(scores1, scores2, both_normal)
            
            # Confidence interval for difference
            results['confidence_interval'] = self.calculate_confidence_interval(
                scores1, scores2, both_normal
            )
            
            # Interpretation
            if results['significant']:
                better_model = model1_name if results['difference'] < 0 else model2_name
                results['interpretation'] = (
                    f"{better_model} significantly outperforms the other "
                    f"(p = {p_value:.4f})"
                )
            else:
                results['interpretation'] = (
                    f"No significant difference between models "
                    f"(p = {p_value:.4f})"
                )
            
        except Exception as e:
            results['error'] = f"Statistical test failed: {e}"
        
        return results
    
    def calculate_effect_size(self, scores1: np.ndarray, scores2: np.ndarray,
                            parametric: bool = True) -> Dict[str, float]:
        """Calculate effect size measures."""
        effect_size = {}
        
        try:
            # Cohen's d (for parametric data)
            if parametric:
                diff = scores1 - scores2
                pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
                cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                effect_size['cohens_d'] = cohens_d
                
                # Interpret Cohen's d
                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    effect_size['cohens_d_interpretation'] = 'negligible'
                elif abs_d < 0.5:
                    effect_size['cohens_d_interpretation'] = 'small'
                elif abs_d < 0.8:
                    effect_size['cohens_d_interpretation'] = 'medium'
                else:
                    effect_size['cohens_d_interpretation'] = 'large'
            
            # Common Language Effect Size (CLES)
            n1, n2 = len(scores1), len(scores2)
            rank_sum = 0
            for score1 in scores1:
                rank_sum += np.sum(scores2 < score1)  # For lower-is-better metrics
            cles = rank_sum / (n1 * n2)
            effect_size['cles'] = cles
            
            # Relative improvement
            mean1, mean2 = np.mean(scores1), np.mean(scores2)
            if mean2 != 0:
                relative_improvement = (mean2 - mean1) / mean2 * 100
                effect_size['relative_improvement_pct'] = relative_improvement
            
        except Exception as e:
            effect_size['error'] = f"Effect size calculation failed: {e}"
        
        return effect_size
    
    def calculate_confidence_interval(self, scores1: np.ndarray, scores2: np.ndarray,
                                    parametric: bool = True) -> Dict[str, float]:
        """Calculate confidence interval for the difference."""
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available"}
        
        ci = {}
        
        try:
            diff = scores1 - scores2
            n = len(diff)
            mean_diff = np.mean(diff)
            
            if parametric and n > 1:
                # t-distribution confidence interval
                std_diff = np.std(diff, ddof=1)
                se_diff = std_diff / np.sqrt(n)
                t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
                margin_error = t_critical * se_diff
                
                ci['lower'] = mean_diff - margin_error
                ci['upper'] = mean_diff + margin_error
                ci['method'] = 't-distribution'
            else:
                # Bootstrap confidence interval
                n_bootstrap = 1000
                bootstrap_diffs = []
                
                for _ in range(n_bootstrap):
                    indices = np.random.choice(n, n, replace=True)
                    bootstrap_diff = np.mean(diff[indices])
                    bootstrap_diffs.append(bootstrap_diff)
                
                lower_percentile = (1 - self.confidence_level) / 2 * 100
                upper_percentile = (1 + self.confidence_level) / 2 * 100
                
                ci['lower'] = np.percentile(bootstrap_diffs, lower_percentile)
                ci['upper'] = np.percentile(bootstrap_diffs, upper_percentile)
                ci['method'] = 'bootstrap'
            
            ci['mean_difference'] = mean_diff
            ci['confidence_level'] = self.confidence_level
            
        except Exception as e:
            ci['error'] = f"Confidence interval calculation failed: {e}"
        
        return ci
    
    def multiple_comparisons(self, all_results: Dict[str, Dict[str, Any]],
                           metric: str = 'test_mae') -> Dict[str, Any]:
        """
        Perform multiple comparisons analysis across all models.
        
        Args:
            all_results: Dictionary with all model results
            metric: Metric to compare
            
        Returns:
            Dictionary with multiple comparisons results
        """
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available"}
        
        print(f"\n🔍 Performing Multiple Comparisons Analysis...")
        print(f"📊 Metric: {metric}")
        
        # Extract scores for each model
        model_scores = {}
        for phase_name, phase_results in all_results.items():
            for model_name, model_results in phase_results.items():
                full_name = f"{phase_name}_{model_name}"
                if 'cv_scores' in model_results and metric in model_results['cv_scores']:
                    scores = np.array(model_results['cv_scores'][metric])
                    model_scores[full_name] = scores
        
        if len(model_scores) < 2:
            return {"error": "Need at least 2 models for comparison"}
        
        results = {
            'metric': metric,
            'n_models': len(model_scores),
            'model_names': list(model_scores.keys())
        }
        
        # Overall test (Friedman test for multiple related samples)
        try:
            score_arrays = list(model_scores.values())
            if len(set(len(scores) for scores in score_arrays)) == 1:
                # All arrays have same length - can use Friedman test
                stat, p_value = friedmanchisquare(*score_arrays)
                results['overall_test'] = {
                    'test': 'Friedman Chi-square',
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            else:
                results['overall_test'] = {
                    'error': 'Unequal sample sizes - cannot perform Friedman test'
                }
        except Exception as e:
            results['overall_test'] = {'error': f"Overall test failed: {e}"}
        
        # Pairwise comparisons
        pairwise_results = []
        p_values = []
        model_names = list(model_scores.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                scores1, scores2 = model_scores[model1], model_scores[model2]
                
                # Only compare if both have same length
                min_length = min(len(scores1), len(scores2))
                if min_length > 0:
                    comparison = self.compare_two_models(
                        scores1[:min_length], scores2[:min_length],
                        model1, model2, metric
                    )
                    pairwise_results.append(comparison)
                    if 'p_value' in comparison:
                        p_values.append(comparison['p_value'])
        
        results['pairwise_comparisons'] = pairwise_results
        
        # Multiple comparisons correction
        if p_values and STATSMODELS_AVAILABLE:
            try:
                rejected, p_corrected, _, _ = multipletests(
                    p_values, alpha=self.alpha, method='holm'
                )
                
                results['multiple_comparisons_correction'] = {
                    'method': 'Holm-Bonferroni',
                    'original_alpha': self.alpha,
                    'n_comparisons': len(p_values),
                    'corrected_p_values': p_corrected.tolist(),
                    'significant_after_correction': rejected.tolist()
                }
                
                # Update pairwise results with corrected p-values
                for i, comparison in enumerate(pairwise_results):
                    if i < len(p_corrected):
                        comparison['p_value_corrected'] = p_corrected[i]
                        comparison['significant_corrected'] = rejected[i]
                        
            except Exception as e:
                results['multiple_comparisons_correction'] = {
                    'error': f"Correction failed: {e}"
                }
        
        return results
    
    def clinical_significance_analysis(self, all_results: Dict[str, Dict[str, Any]],
                                     clinical_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Analyze clinical significance of model predictions.
        
        Args:
            all_results: Dictionary with all model results
            clinical_threshold: Threshold for clinically meaningful difference
            
        Returns:
            Dictionary with clinical significance analysis
        """
        print(f"\n🏥 Clinical Significance Analysis...")
        print(f"🎯 Clinical threshold: ±{clinical_threshold} BDI-II points")
        
        results = {
            'clinical_threshold': clinical_threshold,
            'model_analysis': {}
        }
        
        for phase_name, phase_results in all_results.items():
            for model_name, model_results in phase_results.items():
                full_name = f"{phase_name}_{model_name}"
                
                # Extract MAE scores
                if 'cv_scores' in model_results and 'test_mae' in model_results['cv_scores']:
                    mae_scores = np.array(model_results['cv_scores']['test_mae'])
                    
                    # Clinical acceptability analysis
                    clinically_acceptable = np.mean(mae_scores <= clinical_threshold) * 100
                    
                    # Confidence interval for clinical acceptability
                    n = len(mae_scores)
                    p = clinically_acceptable / 100
                    if n > 0 and 0 < p < 1:
                        se = np.sqrt(p * (1 - p) / n)
                        z_critical = stats.norm.ppf(1 - self.alpha/2)
                        ci_lower = max(0, (p - z_critical * se)) * 100
                        ci_upper = min(1, (p + z_critical * se)) * 100
                    else:
                        ci_lower = ci_upper = clinically_acceptable
                    
                    results['model_analysis'][full_name] = {
                        'mean_mae': np.mean(mae_scores),
                        'std_mae': np.std(mae_scores, ddof=1),
                        'clinical_acceptability_pct': clinically_acceptable,
                        'clinical_acceptability_ci': [ci_lower, ci_upper],
                        'always_clinically_acceptable': np.all(mae_scores <= clinical_threshold),
                        'never_clinically_acceptable': np.all(mae_scores > clinical_threshold)
                    }
        
        # Rank models by clinical acceptability
        if results['model_analysis']:
            ranked_models = sorted(
                results['model_analysis'].items(),
                key=lambda x: x[1]['clinical_acceptability_pct'],
                reverse=True
            )
            results['ranking'] = [model for model, _ in ranked_models]
            
            # Best clinically performing model
            best_model = ranked_models[0]
            results['best_clinical_model'] = {
                'name': best_model[0],
                'clinical_acceptability_pct': best_model[1]['clinical_acceptability_pct'],
                'mean_mae': best_model[1]['mean_mae']
            }
        
        return results
    
    def power_analysis(self, effect_size: float, sample_size: int,
                      alpha: float = None) -> Dict[str, Any]:
        """
        Perform power analysis for the study.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            Dictionary with power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        results = {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha
        }
        
        if not STATSMODELS_AVAILABLE:
            results['error'] = "Statsmodels not available for power analysis"
            return results
        
        try:
            # Calculate power
            power = ttest_power(effect_size, sample_size, alpha, alternative='two-sided')
            results['power'] = power
            
            # Power interpretation
            if power >= 0.8:
                results['power_interpretation'] = 'Adequate (≥0.8)'
            elif power >= 0.6:
                results['power_interpretation'] = 'Moderate (0.6-0.79)'
            else:
                results['power_interpretation'] = 'Low (<0.6)'
            
            # Sample size recommendations
            for target_power in [0.8, 0.9]:
                try:
                    from statsmodels.stats.power import tt_solve_power
                    recommended_n = tt_solve_power(
                        effect_size=effect_size,
                        power=target_power,
                        alpha=alpha,
                        alternative='two-sided'
                    )
                    results[f'recommended_n_for_power_{target_power}'] = int(np.ceil(recommended_n))
                except:
                    pass
            
        except Exception as e:
            results['error'] = f"Power analysis failed: {e}"
        
        return results
    
    def generate_statistical_report(self, all_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive statistical analysis report."""
        
        print("\n📋 Generating Statistical Analysis Report...")
        
        report = []
        report.append("📊 COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Multiple comparisons analysis
        mc_results = self.multiple_comparisons(all_results, 'test_mae')
        if 'error' not in mc_results:
            report.append("🔍 MULTIPLE COMPARISONS ANALYSIS:")
            report.append("-" * 35)
            report.append(f"• Total models compared: {mc_results['n_models']}")
            
            if 'overall_test' in mc_results and 'p_value' in mc_results['overall_test']:
                overall_p = mc_results['overall_test']['p_value']
                report.append(f"• Friedman test p-value: {overall_p:.4f}")
                if overall_p < self.alpha:
                    report.append("• Overall difference: SIGNIFICANT")
                else:
                    report.append("• Overall difference: NOT SIGNIFICANT")
            
            # Significant pairwise comparisons
            significant_pairs = []
            for comp in mc_results.get('pairwise_comparisons', []):
                if comp.get('significant', False):
                    significant_pairs.append(
                        f"{comp['model1']} vs {comp['model2']} (p={comp['p_value']:.4f})"
                    )
            
            if significant_pairs:
                report.append(f"• Significant pairwise differences: {len(significant_pairs)}")
                for pair in significant_pairs[:5]:  # Show top 5
                    report.append(f"  - {pair}")
            else:
                report.append("• No significant pairwise differences found")
            
            report.append("")
        
        # Clinical significance analysis
        clinical_results = self.clinical_significance_analysis(all_results)
        if 'best_clinical_model' in clinical_results:
            report.append("🏥 CLINICAL SIGNIFICANCE ANALYSIS:")
            report.append("-" * 35)
            
            best_clinical = clinical_results['best_clinical_model']
            report.append(f"• Clinical threshold: ±{clinical_results['clinical_threshold']} points")
            report.append(f"• Best clinical model: {best_clinical['name']}")
            report.append(f"• Clinical acceptability: {best_clinical['clinical_acceptability_pct']:.1f}%")
            report.append(f"• Mean MAE: {best_clinical['mean_mae']:.3f}")
            
            # Count models with good clinical performance
            good_clinical = sum(1 for model_data in clinical_results['model_analysis'].values()
                              if model_data['clinical_acceptability_pct'] >= 80)
            report.append(f"• Models with ≥80% clinical acceptability: {good_clinical}")
            report.append("")
        
        # Power analysis (using typical effect size)
        if all_results:
            # Estimate effect size from data
            sample_sizes = []
            for phase_results in all_results.values():
                for model_results in phase_results.values():
                    if 'cv_scores' in model_results and 'test_mae' in model_results['cv_scores']:
                        sample_sizes.append(len(model_results['cv_scores']['test_mae']))
            
            if sample_sizes:
                avg_sample_size = int(np.mean(sample_sizes))
                
                # Power analysis for different effect sizes
                report.append("⚡ POWER ANALYSIS:")
                report.append("-" * 15)
                report.append(f"• Average sample size: {avg_sample_size}")
                
                for effect_size, description in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
                    power_results = self.power_analysis(effect_size, avg_sample_size)
                    if 'power' in power_results:
                        power = power_results['power']
                        interpretation = power_results['power_interpretation']
                        report.append(f"• Power for {description} effect (d={effect_size}): {power:.3f} ({interpretation})")
                
                report.append("")
        
        # Summary and recommendations
        report.append("💡 STATISTICAL SUMMARY & RECOMMENDATIONS:")
        report.append("-" * 42)
        
        # Model selection recommendation
        best_models = []
        for phase_name, phase_results in all_results.items():
            phase_best = min(phase_results.items(), 
                           key=lambda x: x[1]['mean_scores']['test_mae'])
            best_models.append((f"{phase_name}_{phase_best[0]}", phase_best[1]['mean_scores']['test_mae']))
        
        if best_models:
            overall_best = min(best_models, key=lambda x: x[1])
            report.append(f"• Statistically best model: {overall_best[0]} (MAE: {overall_best[1]:.3f})")
        
        # Clinical recommendation
        if 'best_clinical_model' in clinical_results:
            clinical_best = clinical_results['best_clinical_model']
            report.append(f"• Clinically best model: {clinical_best['name']}")
            
            # Check if statistical and clinical best are the same
            if best_models and overall_best[0] == clinical_best['name']:
                report.append("• ✅ Statistical and clinical best models agree")
            else:
                report.append("• ⚠️ Statistical and clinical best models differ")
        
        # Publication recommendations
        report.append("")
        report.append("📝 PUBLICATION RECOMMENDATIONS:")
        report.append("-" * 30)
        report.append("• Report effect sizes alongside p-values")
        report.append("• Include confidence intervals for all estimates")
        report.append("• Consider clinical significance in interpretation")
        report.append("• Apply multiple comparisons correction")
        report.append("• Discuss practical significance vs statistical significance")
        
        return "\n".join(report)
    
    def create_statistical_visualizations(self, all_results: Dict[str, Dict[str, Any]],
                                        save_plots: bool = True, output_dir: str = "../Results") -> None:
        """Create comprehensive statistical visualizations."""
        
        print("\n📊 Creating Statistical Visualizations...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        if save_plots:
            output_path = Path(output_dir) / "Statistical_Analysis"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data for visualization
        model_names = []
        mae_scores = []
        r2_scores = []
        phases = []
        
        for phase_name, phase_results in all_results.items():
            for model_name, model_results in phase_results.items():
                if 'cv_scores' in model_results:
                    model_names.append(f"{phase_name}_{model_name}")
                    mae_scores.append(model_results['cv_scores']['test_mae'])
                    r2_scores.append(model_results['cv_scores']['test_r2'])
                    phases.extend([phase_name] * len(model_results['cv_scores']['test_mae']))
        
        if not model_names:
            print("⚠️ No data available for visualization")
            return
        
        # 1. Model Performance Comparison (Box Plot)
        plt.figure(figsize=(16, 10))
        
        # Prepare data for boxplot
        all_mae = []
        all_models = []
        for i, (name, scores) in enumerate(zip(model_names, mae_scores)):
            all_mae.extend(scores)
            all_models.extend([name] * len(scores))
        
        df_plot = pd.DataFrame({'Model': all_models, 'MAE': all_mae})
        
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df_plot, x='MAE', y='Model', orient='h')
        plt.title('Model Performance Comparison (MAE)', fontsize=14, fontweight='bold')
        plt.xlabel('Mean Absolute Error')
        
        # 2. Effect Size Heatmap
        plt.subplot(2, 2, 2)
        
        # Calculate pairwise effect sizes
        n_models = len(model_names)
        effect_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i != j and len(mae_scores[i]) == len(mae_scores[j]):
                    # Calculate Cohen's d
                    scores1, scores2 = np.array(mae_scores[i]), np.array(mae_scores[j])
                    diff = scores1 - scores2
                    pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
                    if pooled_std > 0:
                        effect_matrix[i, j] = np.mean(diff) / pooled_std
        
        # Create heatmap with abbreviated model names
        abbreviated_names = [name.split('_')[-1][:8] for name in model_names]
        sns.heatmap(effect_matrix, 
                   xticklabels=abbreviated_names,
                   yticklabels=abbreviated_names,
                   center=0, cmap='RdBu_r', 
                   annot=True, fmt='.2f')
        plt.title('Effect Size Matrix (Cohen\'s d)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 3. Performance by Phase
        plt.subplot(2, 2, 3)
        
        # Calculate mean performance by phase
        phase_performance = {}
        for phase_name, phase_results in all_results.items():
            phase_maes = []
            for model_results in phase_results.values():
                if 'mean_scores' in model_results:
                    phase_maes.append(model_results['mean_scores']['test_mae'])
            
            if phase_maes:
                phase_performance[phase_name] = {
                    'mean': np.mean(phase_maes),
                    'std': np.std(phase_maes, ddof=1) if len(phase_maes) > 1 else 0,
                    'min': np.min(phase_maes),
                    'max': np.max(phase_maes)
                }
        
        if phase_performance:
            phases_list = list(phase_performance.keys())
            means = [phase_performance[p]['mean'] for p in phases_list]
            stds = [phase_performance[p]['std'] for p in phases_list]
            
            bars = plt.bar(phases_list, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Average Performance by Phase', fontsize=14, fontweight='bold')
            plt.ylabel('Mean Absolute Error')
            plt.xlabel('Phase')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom')
        
        # 4. Clinical Significance Analysis
        plt.subplot(2, 2, 4)
        
        clinical_results = self.clinical_significance_analysis(all_results)
        if 'model_analysis' in clinical_results:
            clinical_data = clinical_results['model_analysis']
            
            model_names_clin = list(clinical_data.keys())
            clinical_acc = [clinical_data[name]['clinical_acceptability_pct'] 
                          for name in model_names_clin]
            
            # Show only top 10 models for readability
            if len(model_names_clin) > 10:
                sorted_indices = np.argsort(clinical_acc)[-10:]
                model_names_clin = [model_names_clin[i] for i in sorted_indices]
                clinical_acc = [clinical_acc[i] for i in sorted_indices]
            
            abbreviated_names_clin = [name.split('_')[-1][:10] for name in model_names_clin]
            
            bars = plt.barh(abbreviated_names_clin, clinical_acc, alpha=0.7)
            plt.title('Clinical Acceptability (≤3 points MAE)', fontsize=14, fontweight='bold')
            plt.xlabel('Percentage of CV folds with MAE ≤ 3')
            
            # Add value labels
            for bar, acc in zip(bars, clinical_acc):
                width = bar.get_width()
                plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{acc:.1f}%', ha='left', va='center')
            
            plt.xlim(0, 105)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_path / "statistical_analysis_summary.png", 
                       dpi=300, bbox_inches='tight')
            print(f"📊 Statistical visualizations saved to: {output_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalysis()
    print("🚀 Statistical Analysis Module Ready!")