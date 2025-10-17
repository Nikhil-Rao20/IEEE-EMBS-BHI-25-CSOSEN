"""
Therapy Engagement Analysis by Medical Condition
=================================================
Analyzes the importance of therapy-related features (mindfulness sessions started,
completed, therapy completion rate, therapy engagement) for each medical condition
and their impact on BDI scores at 12W and 24W.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "Track1_Data" / "processed"
RESULTS_PATH = BASE_PATH / "Results_12W" / "Therapy_Engagement_Analysis"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("THERAPY ENGAGEMENT BY MEDICAL CONDITION ANALYSIS")
print("="*80)

# Load the dataset
print("\n📊 Loading dataset...")
train_file = DATA_PATH / "train_corrected_features.xlsx"
df = pd.read_excel(train_file)
print(f"✅ Loaded data: {df.shape}")

# Define therapy-related features
therapy_features = [
    'mindfulness_therapies_started',
    'mindfulness_therapies_completed',
    'therapy_completion_rate',
    'therapy_engagement'
]

# Define medical conditions
condition_columns = [
    'condition_cancer',
    'condition_acute_coronary_syndrome',
    'condition_renal_insufficiency',
    'condition_lower_limb_amputation'
]

condition_names = {
    'condition_cancer': 'Cancer',
    'condition_acute_coronary_syndrome': 'Acute Coronary Syndrome',
    'condition_renal_insufficiency': 'Renal Insufficiency',
    'condition_lower_limb_amputation': 'Lower Limb Amputation'
}

# Target variables
target_12w = 'bdi_ii_after_intervention_12w'
target_24w = 'bdi_ii_follow_up_24w'

print(f"\n🔍 Analyzing therapy features: {therapy_features}")
print(f"🏥 Medical conditions: {list(condition_names.values())}")

# Check if therapy features exist
existing_therapy_features = [f for f in therapy_features if f in df.columns]
print(f"\n✓ Available therapy features: {existing_therapy_features}")

# Initialize results storage
results = {
    'analysis_info': {
        'date': pd.Timestamp.now().isoformat(),
        'n_patients': len(df),
        'therapy_features': existing_therapy_features
    },
    'condition_therapy_stats': {},
    'correlations_12w': {},
    'correlations_24w': {},
    'statistical_tests': {}
}

# ============================================================================
# PART 1: Descriptive Statistics - Therapy Features by Condition
# ============================================================================
print("\n" + "="*80)
print("PART 1: THERAPY ENGAGEMENT DESCRIPTIVE STATISTICS BY CONDITION")
print("="*80)

for cond_col, cond_name in condition_names.items():
    if cond_col not in df.columns:
        print(f"⚠️  {cond_name}: Column not found, skipping...")
        continue
    
    with_condition = df[df[cond_col] == 1]
    without_condition = df[df[cond_col] == 0]
    
    n_with = len(with_condition)
    n_without = len(without_condition)
    
    print(f"\n{'─'*80}")
    print(f"🏥 {cond_name.upper()}")
    print(f"{'─'*80}")
    print(f"Patients with condition: {n_with} ({n_with/len(df)*100:.1f}%)")
    print(f"Patients without condition: {n_without} ({n_without/len(df)*100:.1f}%)")
    
    condition_stats = {
        'n_with': int(n_with),
        'n_without': int(n_without),
        'prevalence_pct': float(n_with/len(df)*100),
        'therapy_metrics': {}
    }
    
    print(f"\n{'Therapy Feature':<35} {'With Condition':<20} {'Without Condition':<20} {'Difference':<12} {'p-value':<10}")
    print("─"*100)
    
    for feature in existing_therapy_features:
        with_vals = with_condition[feature].dropna()
        without_vals = without_condition[feature].dropna()
        
        with_mean = with_vals.mean()
        with_std = with_vals.std()
        without_mean = without_vals.mean()
        without_std = without_vals.std()
        
        diff = with_mean - without_mean
        
        # Mann-Whitney U test (non-parametric)
        if len(with_vals) > 0 and len(without_vals) > 0:
            try:
                statistic, p_value = stats.mannwhitneyu(with_vals, without_vals, alternative='two-sided')
            except:
                p_value = np.nan
        else:
            p_value = np.nan
        
        # Cohen's d effect size
        if with_std > 0 or without_std > 0:
            pooled_std = np.sqrt(((len(with_vals)-1)*with_std**2 + (len(without_vals)-1)*without_std**2) / 
                                (len(with_vals) + len(without_vals) - 2))
            cohens_d = diff / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        print(f"{feature:<35} {with_mean:>6.2f} ± {with_std:>5.2f}    {without_mean:>6.2f} ± {without_std:>5.2f}    {diff:>+7.2f}       {p_value:>8.4f}")
        
        condition_stats['therapy_metrics'][feature] = {
            'with_mean': float(with_mean),
            'with_std': float(with_std),
            'without_mean': float(without_mean),
            'without_std': float(without_std),
            'difference': float(diff),
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'cohens_d': float(cohens_d),
            'n_with': int(len(with_vals)),
            'n_without': int(len(without_vals))
        }
    
    results['condition_therapy_stats'][cond_name] = condition_stats

# ============================================================================
# PART 2: Correlation Analysis - Therapy Features vs BDI Outcomes
# ============================================================================
print("\n\n" + "="*80)
print("PART 2: THERAPY-BDI CORRELATION ANALYSIS BY CONDITION")
print("="*80)

for cond_col, cond_name in condition_names.items():
    if cond_col not in df.columns:
        continue
    
    with_condition = df[df[cond_col] == 1]
    
    if len(with_condition) < 5:  # Need minimum sample size
        print(f"\n⚠️  {cond_name}: Insufficient sample size (n={len(with_condition)}), skipping correlation analysis")
        continue
    
    print(f"\n{'─'*80}")
    print(f"🏥 {cond_name.upper()} (n={len(with_condition)})")
    print(f"{'─'*80}")
    
    # 12-week correlations
    print(f"\n📊 12-WEEK BDI CORRELATIONS:")
    print(f"{'Therapy Feature':<35} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} {'p-value':<10}")
    print("─"*85)
    
    corr_12w = {}
    for feature in existing_therapy_features:
        valid_data = with_condition[[feature, target_12w]].dropna()
        
        if len(valid_data) >= 5:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(valid_data[feature], valid_data[target_12w])
            # Spearman correlation (non-parametric)
            spearman_r, spearman_p = stats.spearmanr(valid_data[feature], valid_data[target_12w])
            
            print(f"{feature:<35} {pearson_r:>+8.3f}     {pearson_p:>8.4f}    {spearman_r:>+8.3f}     {spearman_p:>8.4f}")
            
            corr_12w[feature] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'n_samples': int(len(valid_data))
            }
        else:
            print(f"{feature:<35} {'Insufficient data'}")
            corr_12w[feature] = None
    
    # 24-week correlations
    print(f"\n📊 24-WEEK BDI CORRELATIONS:")
    print(f"{'Therapy Feature':<35} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} {'p-value':<10}")
    print("─"*85)
    
    corr_24w = {}
    for feature in existing_therapy_features:
        valid_data = with_condition[[feature, target_24w]].dropna()
        
        if len(valid_data) >= 5:
            pearson_r, pearson_p = stats.pearsonr(valid_data[feature], valid_data[target_24w])
            spearman_r, spearman_p = stats.spearmanr(valid_data[feature], valid_data[target_24w])
            
            print(f"{feature:<35} {pearson_r:>+8.3f}     {pearson_p:>8.4f}    {spearman_r:>+8.3f}     {spearman_p:>8.4f}")
            
            corr_24w[feature] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'n_samples': int(len(valid_data))
            }
        else:
            print(f"{feature:<35} {'Insufficient data'}")
            corr_24w[feature] = None
    
    results['correlations_12w'][cond_name] = corr_12w
    results['correlations_24w'][cond_name] = corr_24w

# ============================================================================
# PART 3: Stratified Analysis - High vs Low Therapy Engagement
# ============================================================================
print("\n\n" + "="*80)
print("PART 3: HIGH vs LOW THERAPY ENGAGEMENT IMPACT BY CONDITION")
print("="*80)

# Use therapy_completion_rate or therapy_engagement as main metric
main_therapy_metric = 'therapy_completion_rate' if 'therapy_completion_rate' in existing_therapy_features else existing_therapy_features[0]

for cond_col, cond_name in condition_names.items():
    if cond_col not in df.columns:
        continue
    
    with_condition = df[df[cond_col] == 1].copy()
    
    if len(with_condition) < 10:  # Need reasonable sample size
        print(f"\n⚠️  {cond_name}: Insufficient sample size (n={len(with_condition)}), skipping stratification")
        continue
    
    # Split by median engagement
    median_engagement = with_condition[main_therapy_metric].median()
    high_engagement = with_condition[with_condition[main_therapy_metric] >= median_engagement]
    low_engagement = with_condition[with_condition[main_therapy_metric] < median_engagement]
    
    print(f"\n{'─'*80}")
    print(f"🏥 {cond_name.upper()}")
    print(f"{'─'*80}")
    print(f"Using therapy metric: {main_therapy_metric}")
    print(f"Median value: {median_engagement:.2f}")
    print(f"High engagement: n={len(high_engagement)}")
    print(f"Low engagement: n={len(low_engagement)}")
    
    # 12-week comparison
    high_12w = high_engagement[target_12w].dropna()
    low_12w = low_engagement[target_12w].dropna()
    
    if len(high_12w) > 0 and len(low_12w) > 0:
        high_mean_12w = high_12w.mean()
        low_mean_12w = low_12w.mean()
        diff_12w = high_mean_12w - low_mean_12w
        
        _, p_12w = stats.mannwhitneyu(high_12w, low_12w, alternative='two-sided')
        
        print(f"\n12-WEEK BDI OUTCOMES:")
        print(f"  High engagement: {high_mean_12w:.2f} ± {high_12w.std():.2f}")
        print(f"  Low engagement:  {low_mean_12w:.2f} ± {low_12w.std():.2f}")
        print(f"  Difference: {diff_12w:+.2f} (p={p_12w:.4f}) {'***' if p_12w < 0.001 else '**' if p_12w < 0.01 else '*' if p_12w < 0.05 else ''}")
    
    # 24-week comparison
    high_24w = high_engagement[target_24w].dropna()
    low_24w = low_engagement[target_24w].dropna()
    
    if len(high_24w) > 0 and len(low_24w) > 0:
        high_mean_24w = high_24w.mean()
        low_mean_24w = low_24w.mean()
        diff_24w = high_mean_24w - low_mean_24w
        
        _, p_24w = stats.mannwhitneyu(high_24w, low_24w, alternative='two-sided')
        
        print(f"\n24-WEEK BDI OUTCOMES:")
        print(f"  High engagement: {high_mean_24w:.2f} ± {high_24w.std():.2f}")
        print(f"  Low engagement:  {low_mean_24w:.2f} ± {low_24w.std():.2f}")
        print(f"  Difference: {diff_24w:+.2f} (p={p_24w:.4f}) {'***' if p_24w < 0.001 else '**' if p_24w < 0.01 else '*' if p_24w < 0.05 else ''}")

# ============================================================================
# VISUALIZATION: Correlation Heatmaps
# ============================================================================
print("\n\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create correlation heatmap for 12W
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 12W correlations
corr_data_12w = []
for cond_name in condition_names.values():
    if cond_name in results['correlations_12w']:
        row = []
        for feature in existing_therapy_features:
            if results['correlations_12w'][cond_name].get(feature):
                row.append(results['correlations_12w'][cond_name][feature]['pearson_r'])
            else:
                row.append(0)
        corr_data_12w.append(row)

if corr_data_12w:
    corr_df_12w = pd.DataFrame(corr_data_12w, 
                               index=[c for c in condition_names.values() if c in results['correlations_12w']], 
                               columns=existing_therapy_features)
    
    sns.heatmap(corr_df_12w, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                vmin=-0.5, vmax=0.5, ax=axes[0], cbar_kws={'label': 'Pearson r'})
    axes[0].set_title('Therapy Features vs 12-Week BDI Score Correlation by Medical Condition', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Therapy Features', fontsize=11)
    axes[0].set_ylabel('Medical Condition', fontsize=11)

# 24W correlations
corr_data_24w = []
for cond_name in condition_names.values():
    if cond_name in results['correlations_24w']:
        row = []
        for feature in existing_therapy_features:
            if results['correlations_24w'][cond_name].get(feature):
                row.append(results['correlations_24w'][cond_name][feature]['pearson_r'])
            else:
                row.append(0)
        corr_data_24w.append(row)

if corr_data_24w:
    corr_df_24w = pd.DataFrame(corr_data_24w, 
                               index=[c for c in condition_names.values() if c in results['correlations_24w']], 
                               columns=existing_therapy_features)
    
    sns.heatmap(corr_df_24w, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                vmin=-0.5, vmax=0.5, ax=axes[1], cbar_kws={'label': 'Pearson r'})
    axes[1].set_title('Therapy Features vs 24-Week BDI Score Correlation by Medical Condition', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Therapy Features', fontsize=11)
    axes[1].set_ylabel('Medical Condition', fontsize=11)

plt.tight_layout()
heatmap_file = RESULTS_PATH / "therapy_engagement_correlation_heatmap.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved correlation heatmap: {heatmap_file}")
plt.close()

# Create bar chart comparing therapy engagement across conditions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(existing_therapy_features):
    means_with = []
    stds_with = []
    conditions_list = []
    
    for cond_name in condition_names.values():
        if cond_name in results['condition_therapy_stats']:
            if feature in results['condition_therapy_stats'][cond_name]['therapy_metrics']:
                means_with.append(results['condition_therapy_stats'][cond_name]['therapy_metrics'][feature]['with_mean'])
                stds_with.append(results['condition_therapy_stats'][cond_name]['therapy_metrics'][feature]['with_std'])
                conditions_list.append(cond_name)
    
    if means_with:
        x_pos = np.arange(len(conditions_list))
        axes[idx].bar(x_pos, means_with, yerr=stds_with, alpha=0.7, capsize=5, color='steelblue')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(conditions_list, rotation=45, ha='right')
        axes[idx].set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
        axes[idx].set_ylabel('Mean Value')
        axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Therapy Engagement Metrics by Medical Condition', fontsize=16, fontweight='bold')
plt.tight_layout()
barplot_file = RESULTS_PATH / "therapy_engagement_by_condition_barplot.png"
plt.savefig(barplot_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved bar plot: {barplot_file}")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
results_file = RESULTS_PATH / "therapy_engagement_analysis_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved analysis results: {results_file}")

# Create summary table CSV
summary_rows = []
for cond_name in condition_names.values():
    if cond_name in results['condition_therapy_stats']:
        for feature in existing_therapy_features:
            if feature in results['condition_therapy_stats'][cond_name]['therapy_metrics']:
                metrics = results['condition_therapy_stats'][cond_name]['therapy_metrics'][feature]
                
                # Add 12W correlation if available
                corr_12w_r = results['correlations_12w'].get(cond_name, {}).get(feature, {}).get('pearson_r', np.nan)
                corr_12w_p = results['correlations_12w'].get(cond_name, {}).get(feature, {}).get('pearson_p', np.nan)
                
                # Add 24W correlation if available
                corr_24w_r = results['correlations_24w'].get(cond_name, {}).get(feature, {}).get('pearson_r', np.nan)
                corr_24w_p = results['correlations_24w'].get(cond_name, {}).get(feature, {}).get('pearson_p', np.nan)
                
                summary_rows.append({
                    'Condition': cond_name,
                    'Therapy_Feature': feature,
                    'Mean_With_Condition': metrics['with_mean'],
                    'Std_With_Condition': metrics['with_std'],
                    'Mean_Without_Condition': metrics['without_mean'],
                    'Std_Without_Condition': metrics['without_std'],
                    'Difference': metrics['difference'],
                    'P_Value': metrics['p_value'],
                    'Cohens_D': metrics['cohens_d'],
                    'Correlation_12W_r': corr_12w_r,
                    'Correlation_12W_p': corr_12w_p,
                    'Correlation_24W_r': corr_24w_r,
                    'Correlation_24W_p': corr_24w_p
                })

summary_df = pd.DataFrame(summary_rows)
summary_file = RESULTS_PATH / "therapy_engagement_summary_table.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✅ Saved summary table: {summary_file}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults saved to: {RESULTS_PATH}")
print(f"  - JSON results: therapy_engagement_analysis_results.json")
print(f"  - Summary CSV: therapy_engagement_summary_table.csv")
print(f"  - Correlation heatmap: therapy_engagement_correlation_heatmap.png")
print(f"  - Bar plot: therapy_engagement_by_condition_barplot.png")
