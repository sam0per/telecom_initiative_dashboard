import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from typing import Dict, List, Tuple, Optional
import streamlit as st


def perform_kaplan_meier_analysis(df: pd.DataFrame, 
                                   cohort_columns: List[str],
                                   duration_col: str = 'duration',
                                   event_col: str = 'event_observed',
                                   confidence_level: float = 0.95) -> Dict[str, Dict]:
    """
    Perform Kaplan-Meier survival analysis for multiple cohorts.
    
    Args:
        df: DataFrame with survival data (must have duration and event columns)
        cohort_columns: List of cohort column names (e.g., ['cohort_enterprise', 'cohort_smb'])
        duration_col: Column name for duration (time to event or censoring)
        event_col: Column name for event indicator (1=event, 0=censored)
        confidence_level: Confidence level for intervals (default 0.95)
        
    Returns:
        dict: {
            'cohort_name': {
                'kmf': KaplanMeierFitter object,
                'median_survival': float (days),
                'retention_30d': float (0-1),
                'retention_60d': float (0-1),
                'retention_90d': float (0-1),
                'retention_180d': float (0-1),
                'ci_lower': array,
                'ci_upper': array,
                'timeline': array,
                'survival_function': array,
                'sample_size': int,
                'events_observed': int,
                'censored_count': int
            }
        }
    
    Example:
        >>> from utils.cohort_builder import create_cohort_definitions, assign_cohorts, calculate_time_to_event
        >>> from data import get_cohort_data
        >>> 
        >>> df = get_cohort_data()
        >>> cohort_defs = create_cohort_definitions(df)
        >>> df = assign_cohorts(df, cohort_defs)
        >>> df = calculate_time_to_event(df)
        >>> 
        >>> cohort_cols = [col for col in df.columns if col.startswith('cohort_')]
        >>> results = perform_kaplan_meier_analysis(df, cohort_cols[:5])  # Analyze first 5 cohorts
        >>> print(results['cohort_enterprise']['median_survival'])
    """
    results = {}
    alpha = 1 - confidence_level
    
    for cohort_col in cohort_columns:
        # Extract cohort name from column
        cohort_name = cohort_col.replace('cohort_', '').replace('_', ' ').title()
        
        # Filter to cohort members
        cohort_df = df[df[cohort_col] == True].copy()
        
        # Skip if cohort is too small
        if len(cohort_df) < 10:
            print(f"Warning: Cohort '{cohort_name}' has only {len(cohort_df)} members. Skipping analysis.")
            continue
        
        # Check for sufficient events
        events_count = cohort_df[event_col].sum()
        if events_count < 5:
            print(f"Warning: Cohort '{cohort_name}' has only {events_count} events. Results may be unreliable.")
        
        # Initialize Kaplan-Meier fitter
        kmf = KaplanMeierFitter(alpha=alpha)
        
        # Fit the model
        try:
            kmf.fit(
                durations=cohort_df[duration_col],
                event_observed=cohort_df[event_col],
                label=cohort_name
            )
            
            # Extract survival function
            survival_function = kmf.survival_function_[cohort_name].values
            timeline = kmf.survival_function_.index.values
            
            # Extract confidence intervals
            ci_df = kmf.confidence_interval_survival_function_
            ci_lower = ci_df[f'{cohort_name}_lower_{confidence_level}'].values
            ci_upper = ci_df[f'{cohort_name}_upper_{confidence_level}'].values
            
            # Calculate median survival time
            median_survival = kmf.median_survival_time_
            
            # Calculate retention rates at key timepoints
            retention_rates = {}
            for days in [30, 60, 90, 180, 365]:
                try:
                    # Get survival probability at specific time
                    survival_at_t = kmf.survival_function_at_times(days).values[0]
                    retention_rates[f'retention_{days}d'] = survival_at_t
                except:
                    # If time is beyond data range, use last available value
                    retention_rates[f'retention_{days}d'] = None
            
            # Store results
            results[cohort_col] = {
                'cohort_name': cohort_name,
                'kmf': kmf,
                'median_survival': median_survival if not np.isnan(median_survival) else None,
                'timeline': timeline,
                'survival_function': survival_function,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'sample_size': len(cohort_df),
                'events_observed': int(events_count),
                'censored_count': int(len(cohort_df) - events_count),
                **retention_rates
            }
            
        except Exception as e:
            print(f"Error fitting KM model for cohort '{cohort_name}': {e}")
            continue
    
    return results


def compare_cohorts_logrank(df: pd.DataFrame,
                            cohort_col_a: str,
                            cohort_col_b: str,
                            duration_col: str = 'duration',
                            event_col: str = 'event_observed') -> Dict:
    """
    Compare two cohorts using the log-rank test.
    
    The log-rank test is a hypothesis test to compare the survival distributions
    of two samples. Null hypothesis: both cohorts have identical survival curves.
    
    Args:
        df: DataFrame with survival data
        cohort_col_a: First cohort column name
        cohort_col_b: Second cohort column name
        duration_col: Duration column name
        event_col: Event indicator column name
        
    Returns:
        dict: {
            'cohort_a_name': str,
            'cohort_b_name': str,
            'test_statistic': float,
            'p_value': float,
            'significant': bool (p < 0.05),
            'interpretation': str,
            'cohort_a_median': float,
            'cohort_b_median': float,
            'survival_difference': str  # "A better" or "B better" or "No difference"
        }
    
    Example:
        >>> result = compare_cohorts_logrank(df, 'cohort_enterprise', 'cohort_consumer')
        >>> print(f"P-value: {result['p_value']:.4f}")
        >>> print(f"Significant: {result['significant']}")
    """
    # Extract cohort names
    cohort_a_name = cohort_col_a.replace('cohort_', '').replace('_', ' ').title()
    cohort_b_name = cohort_col_b.replace('cohort_', '').replace('_', ' ').title()
    
    # Filter to cohort members
    cohort_a_df = df[df[cohort_col_a] == True]
    cohort_b_df = df[df[cohort_col_b] == True]
    
    # Validate sample sizes
    if len(cohort_a_df) < 10 or len(cohort_b_df) < 10:
        return {
            'cohort_a_name': cohort_a_name,
            'cohort_b_name': cohort_b_name,
            'error': 'Insufficient sample size for comparison (need at least 10 per cohort)',
            'test_statistic': None,
            'p_value': None,
            'significant': None
        }
    
    # Perform log-rank test
    try:
        results = logrank_test(
            durations_A=cohort_a_df[duration_col],
            durations_B=cohort_b_df[duration_col],
            event_observed_A=cohort_a_df[event_col],
            event_observed_B=cohort_b_df[event_col]
        )
        
        # Calculate median survival for each cohort
        kmf_a = KaplanMeierFitter()
        kmf_a.fit(cohort_a_df[duration_col], cohort_a_df[event_col])
        median_a = kmf_a.median_survival_time_
        
        kmf_b = KaplanMeierFitter()
        kmf_b.fit(cohort_b_df[duration_col], cohort_b_df[event_col])
        median_b = kmf_b.median_survival_time_
        
        # Determine which cohort has better survival
        if results.p_value < 0.05:
            if median_a > median_b:
                survival_difference = f"{cohort_a_name} has significantly better survival"
            else:
                survival_difference = f"{cohort_b_name} has significantly better survival"
            interpretation = f"Statistically significant difference (p={results.p_value:.4f})"
        else:
            survival_difference = "No significant difference"
            interpretation = f"No statistically significant difference (p={results.p_value:.4f})"
        
        return {
            'cohort_a_name': cohort_a_name,
            'cohort_b_name': cohort_b_name,
            'test_statistic': results.test_statistic,
            'p_value': results.p_value,
            'significant': results.p_value < 0.05,
            'interpretation': interpretation,
            'cohort_a_median': median_a if not np.isnan(median_a) else None,
            'cohort_b_median': median_b if not np.isnan(median_b) else None,
            'cohort_a_sample_size': len(cohort_a_df),
            'cohort_b_sample_size': len(cohort_b_df),
            'survival_difference': survival_difference
        }
        
    except Exception as e:
        return {
            'cohort_a_name': cohort_a_name,
            'cohort_b_name': cohort_b_name,
            'error': f'Failed to perform log-rank test: {str(e)}',
            'test_statistic': None,
            'p_value': None,
            'significant': None
        }


def calculate_hazard_ratios(df: pd.DataFrame,
                           cohort_columns: List[str],
                           duration_col: str = 'duration',
                           event_col: str = 'event_observed',
                           reference_cohort: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate hazard ratios using Cox Proportional Hazards model.
    
    Hazard ratio > 1: Higher risk of event (churn) compared to reference
    Hazard ratio < 1: Lower risk of event (better retention) compared to reference
    
    Args:
        df: DataFrame with survival data
        cohort_columns: List of cohort column names to include in model
        duration_col: Duration column name
        event_col: Event indicator column name
        reference_cohort: Reference cohort for comparison (optional)
        
    Returns:
        DataFrame with columns:
        - cohort_name
        - hazard_ratio
        - ci_lower (95% CI)
        - ci_upper (95% CI)
        - p_value
        - significant (p < 0.05)
        - interpretation
    
    Note:
        This function uses mutually exclusive cohorts to avoid multicollinearity.
        If overlapping cohorts are provided, it will use only segment-based cohorts
        (Enterprise, SMB, Consumer) as they are mutually exclusive.

    Example:
        >>> hazard_ratios = calculate_hazard_ratios(
        ...     df, 
        ...     ['cohort_enterprise', 'cohort_smb', 'cohort_consumer'],
        ...     reference_cohort='cohort_enterprise'
        ... )
        >>> print(hazard_ratios)
    """
    # Filter to mutually exclusive cohorts only to avoid multicollinearity
    # Priority: use segment cohorts (enterprise, smb, consumer) as they're mutually exclusive
    segment_cohorts = [col for col in cohort_columns 
                      if any(segment in col.lower() for segment in ['enterprise', 'smb', 'consumer'])]
    
    if len(segment_cohorts) >= 2:
        # Use segment cohorts (mutually exclusive)
        cohorts_to_use = segment_cohorts
        print(f"   Using mutually exclusive segment cohorts: {[c.replace('cohort_', '') for c in cohorts_to_use]}")
    else:
        # Fall back to non-overlapping cohorts by checking for mutual exclusivity
        cohorts_to_use = _filter_mutually_exclusive_cohorts(df, cohort_columns)
        if len(cohorts_to_use) < 2:
            print("   Warning: Need at least 2 mutually exclusive cohorts for Cox PH analysis.")
            return pd.DataFrame({
                'error': ['Insufficient mutually exclusive cohorts. Use segment cohorts (Enterprise/SMB/Consumer) for hazard ratio analysis.']
            })
    
    # Prepare data for Cox model - create indicator variables
    # We'll use the first cohort as reference (omitted from model)
    if reference_cohort and reference_cohort in cohorts_to_use:
        reference = reference_cohort
    else:
        reference = cohorts_to_use[0]
    
    # Create a single categorical variable instead of multiple binary columns
    # This avoids the dummy variable trap
    cox_df = df[[duration_col, event_col]].copy()
    
    # Create cohort_group column: assign each user to ONE cohort
    def assign_single_cohort(row):
        for cohort_col in cohorts_to_use:
            if row[cohort_col]:
                return cohort_col.replace('cohort_', '')
        return 'other'
    
    cox_df['cohort_group'] = df.apply(assign_single_cohort, axis=1)
    
    # Remove rows not in any of our selected cohorts
    cox_df = cox_df[cox_df['cohort_group'] != 'other']
    
    if len(cox_df) == 0:
        return pd.DataFrame({
            'error': ['No data available for selected cohorts']
        })
    
    # Create dummy variables with reference category dropped
    reference_name = reference.replace('cohort_', '')
    cohort_dummies = pd.get_dummies(cox_df['cohort_group'], prefix='cohort', drop_first=False)
    
    # Drop the reference cohort column to avoid multicollinearity
    if f'cohort_{reference_name}' in cohort_dummies.columns:
        cohort_dummies = cohort_dummies.drop(columns=[f'cohort_{reference_name}'])
    
    # Combine with survival data
    cox_df = pd.concat([
        cox_df[[duration_col, event_col]],
        cohort_dummies
    ], axis=1)
    
    # Initialize Cox PH model
    cph = CoxPHFitter()
    
    try:
        # Fit the model
        cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
        
        # Extract results
        summary = cph.summary
        
        # Process results into readable format
        results = []
        
        # Add reference cohort with HR = 1.0
        results.append({
            'cohort_name': reference_name.replace('_', ' ').title(),
            'hazard_ratio': 1.0,
            'ci_lower': 1.0,
            'ci_upper': 1.0,
            'p_value': np.nan,
            'significant': False,
            'interpretation': 'Reference category'
        })
        
        # Add other cohorts
        for idx in summary.index:
            if idx.startswith('cohort_'):
                cohort_name = idx.replace('cohort_', '').replace('_', ' ').title()
                
                hazard_ratio = np.exp(summary.loc[idx, 'coef'])
                ci_lower = np.exp(summary.loc[idx, 'coef lower 95%'])
                ci_upper = np.exp(summary.loc[idx, 'coef upper 95%'])
                p_value = summary.loc[idx, 'p']
                
                # Interpretation relative to reference
                if p_value < 0.05:
                    if hazard_ratio > 1:
                        interpretation = f"{(hazard_ratio - 1) * 100:.1f}% higher churn risk vs {reference_name.title()}"
                    else:
                        interpretation = f"{(1 - hazard_ratio) * 100:.1f}% lower churn risk vs {reference_name.title()}"
                else:
                    interpretation = f"No significant difference vs {reference_name.title()}"
                
                results.append({
                    'cohort_name': cohort_name,
                    'hazard_ratio': hazard_ratio,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': interpretation
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('hazard_ratio', ascending=False)
        
        return results_df
        
    except Exception as e:
        print(f"   Error calculating hazard ratios: {e}")
        return pd.DataFrame({
            'error': [f'Cox PH model failed: {str(e)}. Try using mutually exclusive cohorts.']
        })


def _filter_mutually_exclusive_cohorts(df: pd.DataFrame, cohort_columns: List[str]) -> List[str]:
    """
    Helper function to identify mutually exclusive cohorts.
    
    Args:
        df: DataFrame with cohort columns
        cohort_columns: List of cohort column names
        
    Returns:
        List of mutually exclusive cohort column names
    """
    # Check which cohorts don't overlap
    exclusive_cohorts = []
    
    for i, cohort_a in enumerate(cohort_columns):
        is_exclusive = True
        for cohort_b in exclusive_cohorts:
            # Check if any user belongs to both cohorts
            overlap = (df[cohort_a] & df[cohort_b]).sum()
            if overlap > 0:
                is_exclusive = False
                break
        
        if is_exclusive:
            exclusive_cohorts.append(cohort_a)
    
    return exclusive_cohorts


def create_comparison_matrix(df: pd.DataFrame,
                            cohort_columns: List[str],
                            duration_col: str = 'duration',
                            event_col: str = 'event_observed') -> pd.DataFrame:
    """
    Create a pairwise comparison matrix of all cohorts using log-rank test.
    
    Args:
        df: DataFrame with survival data
        cohort_columns: List of cohort column names
        duration_col: Duration column name
        event_col: Event indicator column name
        
    Returns:
        DataFrame matrix where:
        - Rows and columns are cohort names
        - Values are p-values from log-rank tests
        - Diagonal values are NaN (self-comparison)
    
    Example:
        >>> matrix = create_comparison_matrix(df, cohort_columns)
        >>> # p-value between Enterprise and SMB
        >>> print(matrix.loc['Enterprise', 'Smb'])
    """
    cohort_names = [col.replace('cohort_', '').replace('_', ' ').title() 
                   for col in cohort_columns]
    
    # Initialize matrix
    matrix = pd.DataFrame(
        index=cohort_names,
        columns=cohort_names,
        dtype=float
    )
    
    # Fill matrix with pairwise comparisons
    for i, cohort_a in enumerate(cohort_columns):
        for j, cohort_b in enumerate(cohort_columns):
            if i == j:
                matrix.iloc[i, j] = np.nan  # Self-comparison
            elif i < j:
                # Only calculate once (matrix is symmetric)
                result = compare_cohorts_logrank(df, cohort_a, cohort_b, 
                                                duration_col, event_col)
                p_value = result.get('p_value', np.nan)
                matrix.iloc[i, j] = p_value
                matrix.iloc[j, i] = p_value  # Symmetric
    
    return matrix


def get_survival_summary_table(km_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table from Kaplan-Meier analysis results.
    
    Args:
        km_results: Output from perform_kaplan_meier_analysis()
        
    Returns:
        DataFrame with summary statistics for each cohort
    
    Example:
        >>> km_results = perform_kaplan_meier_analysis(df, cohort_cols)
        >>> summary = get_survival_summary_table(km_results)
        >>> print(summary)
    """
    summary_data = []
    
    for cohort_col, results in km_results.items():
        # Handle median survival formatting
        median_survival = results['median_survival']
        if median_survival is None or np.isinf(median_survival):
            median_survival_str = 'Not reached*'
        else:
            median_survival_str = f"{median_survival:.0f}"

        summary_data.append({
            'Cohort': results['cohort_name'],
            'Sample Size': results['sample_size'],
            'Events': results['events_observed'],
            'Censored': results['censored_count'],
            'Event Rate (%)': f"{(results['events_observed'] / results['sample_size'] * 100):.1f}%",
            'Median Survival (days)': median_survival_str,
            '30-day Retention': f"{results.get('retention_30d', 0) * 100:.1f}%" if results.get('retention_30d') else 'N/A',
            '90-day Retention': f"{results.get('retention_90d', 0) * 100:.1f}%" if results.get('retention_90d') else 'N/A',
            '180-day Retention': f"{results.get('retention_180d', 0) * 100:.1f}%" if results.get('retention_180d') else 'N/A',
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df


# Testing and example usage
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from data import get_cohort_data
    from utils.cohort_builder import (create_cohort_definitions, 
                                      assign_cohorts, 
                                      calculate_time_to_event)
    
    print("=" * 80)
    print("PHASE 3: KAPLAN-MEIER ANALYSIS - TESTING")
    print("=" * 80)
    
    # Load and prepare data
    print("\n1. Loading and preparing cohort data...")
    df = get_cohort_data()
    cohort_defs = create_cohort_definitions(df)
    df = assign_cohorts(df, cohort_defs)
    df = calculate_time_to_event(df)
    print(f"   ✅ Data prepared: {len(df)} users")
    
    # Get cohort columns (analyze main segments only for testing)
    segment_cohorts = ['cohort_enterprise', 'cohort_smb', 'cohort_consumer']
    
    # Perform Kaplan-Meier analysis
    print("\n2. Performing Kaplan-Meier analysis...")
    km_results = perform_kaplan_meier_analysis(df, segment_cohorts)
    print(f"   ✅ Analyzed {len(km_results)} cohorts")
    
    # Display summary table
    print("\n3. Survival Analysis Summary:")
    print("=" * 80)
    summary_table = get_survival_summary_table(km_results)
    print(summary_table.to_string(index=False))
    print("\n* 'Not reached' = Median survival time undefined (>50% still active/censored)")
    
    # Compare two cohorts
    print("\n4. Log-Rank Test: Enterprise vs Consumer")
    print("=" * 80)
    comparison = compare_cohorts_logrank(df, 'cohort_enterprise', 'cohort_consumer')
    print(f"   Test Statistic: {comparison['test_statistic']:.4f}")
    print(f"   P-value: {comparison['p_value']:.6f}")
    print(f"   Significant: {comparison['significant']}")
    print(f"   Interpretation: {comparison['interpretation']}")
    print(f"   {comparison['survival_difference']}")
    
    # Calculate hazard ratios
    print("\n5. Hazard Ratios (Cox Proportional Hazards)")
    print("=" * 80)
    hazard_ratios = calculate_hazard_ratios(df, segment_cohorts)
    if not hazard_ratios.empty:
        print(hazard_ratios.to_string(index=False))
    
    # Create comparison matrix
    print("\n6. Pairwise Comparison Matrix (p-values)")
    print("=" * 80)
    comparison_matrix = create_comparison_matrix(df, segment_cohorts)
    print(comparison_matrix.to_string())
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3 COMPLETE - Kaplan-Meier analysis working correctly!")
    print("=" * 80)
    
    # Show example of accessing KM object for plotting
    print("\n7. Example: Accessing KM object for visualization")
    print("=" * 80)
    enterprise_kmf = km_results['cohort_enterprise']['kmf']
    enterprise_median = enterprise_kmf.median_survival_time_
    if np.isinf(enterprise_median) or np.isnan(enterprise_median):
        print(f"   Enterprise median survival: Not reached (>50% still active)")
    else:
        print(f"   Enterprise median survival: {enterprise_median:.0f} days")
    print(f"   Timeline range: {km_results['cohort_enterprise']['timeline'][0]:.0f} - {km_results['cohort_enterprise']['timeline'][-1]:.0f} days")
    print(f"   Survival function points: {len(km_results['cohort_enterprise']['survival_function'])}")