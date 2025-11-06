import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from typing import Dict, List, Tuple, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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
                except (IndexError, KeyError, ValueError):
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
    
    Performance:
        This function precomputes a cohort overlap matrix to reduce redundant checks.
        For very large numbers of cohorts, performance may still be O(n²) in the worst case,
        but the overlap checks are vectorized and more efficient than nested loops.
    """
    # Precompute overlap matrix: True if cohorts overlap, False otherwise
    n = len(cohort_columns)
    overlap_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if any user belongs to both cohorts
            overlap = (df[cohort_columns[i]] & df[cohort_columns[j]]).any()
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap

    exclusive_cohorts = []
    exclusive_indices = set()
    for i, cohort_a in enumerate(cohort_columns):
        # Check if cohort_a overlaps with any already included exclusive cohort
        if all(not overlap_matrix[i, j] for j in exclusive_indices):
            exclusive_cohorts.append(cohort_a)
            exclusive_indices.add(i)
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


def plot_survival_curves(km_results: Dict[str, Dict], 
                        title: str = "Kaplan-Meier Survival Curves") -> go.Figure:
    """
    Create interactive Plotly visualization of survival curves.
    
    Args:
        km_results: Output from perform_kaplan_meier_analysis()
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Define color palette
    colors = px.colors.qualitative.Set2
    
    for idx, (cohort_col, results) in enumerate(km_results.items()):
        cohort_name = results['cohort_name']
        timeline = results['timeline']
        survival = results['survival_function']
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
        
        color = colors[idx % len(colors)]
        
        # Main survival curve
        fig.add_trace(go.Scatter(
            x=timeline,
            y=survival,
            name=cohort_name,
            mode='lines',
            line=dict(width=3, color=color),
            hovertemplate=(
                f'<b>{cohort_name}</b><br>' +
                'Day: %{x}<br>' +
                'Survival: %{y:.1%}<br>' +
                '<extra></extra>'
            )
        ))
        
        # Confidence interval upper bound
        fig.add_trace(go.Scatter(
            x=timeline,
            y=ci_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Confidence interval lower bound with fill
        fig.add_trace(go.Scatter(
            x=timeline,
            y=ci_lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba({int(color[4:-1].split(",")[0])}, {int(color[4:-1].split(",")[1])}, {int(color[4:-1].split(",")[2])}, 0.2)',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#1f77b4')
        ),
        xaxis=dict(
            title='Days Since Signup',
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='Survival Probability',
            tickformat='.0%',
            gridcolor='lightgray',
            showgrid=True,
            range=[0, 1.05]
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgray',
            borderwidth=1
        )
    )
    
    return fig


def plot_comparison_heatmap(comparison_matrix: pd.DataFrame,
                           title: str = "Cohort Comparison Matrix (p-values)") -> go.Figure:
    """
    Create heatmap of pairwise comparison p-values.
    
    Args:
        comparison_matrix: Output from create_comparison_matrix()
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    # Create text annotations for p-values
    text_annotations = comparison_matrix.apply(
        lambda row: [f'{x:.4f}' if pd.notna(x) else '-' for x in row], axis=1, result_type='expand'
    )
    
    # Create custom colorscale: green (not significant) to red (significant)
    colorscale = [
        [0.0, '#2ecc71'],    # Green (p > 0.05, not significant)
        [0.05, '#f39c12'],   # Orange (p = 0.05, threshold)
        [1.0, '#e74c3c']     # Red (p < 0.05, significant)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=comparison_matrix.values,
        x=comparison_matrix.columns,
        y=comparison_matrix.index,
        text=text_annotations.values,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale=colorscale,
        zmid=0.05,
        zmin=0,
        zmax=0.1,
        colorbar=dict(
            title='p-value',
            tickvals=[0, 0.05, 0.1],
            ticktext=['0.00', '0.05', '0.10']
        ),
        hovertemplate=(
            'Cohort A: %{y}<br>' +
            'Cohort B: %{x}<br>' +
            'p-value: %{z:.4f}<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18)
        ),
        xaxis=dict(title='', side='bottom'),
        yaxis=dict(title=''),
        height=400,
        width=600
    )
    
    return fig


def display_page():
    """
    Main Streamlit page for Cohort Survival Analysis.
    """
    st.set_page_config(page_title="Cohort Analysis", page_icon="📊", layout="wide")
    
    # Header
    st.title("📊 Cohort Survival Analysis")
    st.markdown("""
    Analyze user retention patterns across different cohorts using **Kaplan-Meier survival curves** 
    and statistical comparison tests.
    """)
    
    # Load data
    with st.spinner("Loading cohort data..."):
        from data import get_cohort_data
        from utils.cohort_builder import (
            create_cohort_definitions,
            assign_cohorts,
            calculate_time_to_event
        )
        
        # Get and prepare data
        df = get_cohort_data()
        cohort_defs = create_cohort_definitions(df)
        df = assign_cohorts(df, cohort_defs)
        df = calculate_time_to_event(df)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Configuration")
    
    # Get all cohort columns
    all_cohort_cols = [col for col in df.columns if col.startswith('cohort_')]
    all_cohort_names = [col.replace('cohort_', '').replace('_', ' ').title() 
                       for col in all_cohort_cols]
    
    # Create cohort name to column mapping
    cohort_mapping = dict(zip(all_cohort_names, all_cohort_cols))
    
    # Cohort selection
    st.sidebar.subheader("📋 Select Cohorts")
    
    # Predefined cohort groups for easy selection
    preset = st.sidebar.radio(
        "Quick Select:",
        ["Custom", "User Segments", "Regional", "Plan Types", "Seasonal"]
    )
    
    if preset == "User Segments":
        default_selection = ['Enterprise', 'Smb', 'Consumer']
    elif preset == "Regional":
        default_selection = ['North Region', 'South Region', 'East Region', 'West Region']
    elif preset == "Plan Types":
        default_selection = ['Premium Plan', 'Standard Plan', 'Basic Plan']
    elif preset == "Seasonal":
        default_selection = ['Q1 Signups', 'Q2 Signups', 'Q3 Signups', 'Q4 Signups']
    else:
        default_selection = ['Enterprise', 'Smb', 'Consumer']
    
    selected_cohort_names = st.sidebar.multiselect(
        "Cohorts to analyze:",
        options=all_cohort_names,
        default=default_selection
    )
    
    if not selected_cohort_names:
        st.warning("⚠️ Please select at least one cohort to analyze.")
        st.stop()
    
    # Convert selected names back to column names
    selected_cohort_cols = [cohort_mapping[name] for name in selected_cohort_names]
    
    # Confidence level
    confidence_level = st.sidebar.slider(
        "Confidence Level:",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        format="%d%%"
    ) / 100
    
    st.sidebar.markdown("---")
    
    # Event type (for future extensibility)
    event_type = st.sidebar.radio(
        "📌 Event Type:",
        ["Churn", "Conversion (Future)"],
        index=0
    )
    
    if event_type == "Conversion (Future)":
        st.sidebar.info("Conversion event analysis coming soon!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **💡 Understanding the Analysis:**
    
    - **Survival Curve**: Shows % of users still active over time
    - **Median Survival**: Time when 50% have churned
    - **Confidence Intervals**: Uncertainty bounds (shaded areas)
    - **Log-Rank Test**: Statistical comparison between cohorts
    - **Hazard Ratio**: Relative churn risk between cohorts
    """)
    
    # Perform analysis
    with st.spinner("Performing Kaplan-Meier analysis..."):
        km_results = perform_kaplan_meier_analysis(
            df, 
            selected_cohort_cols,
            confidence_level=confidence_level
        )
    
    if not km_results:
        st.error("❌ Analysis failed. Please check your cohort selection.")
        st.stop()
    
    # Calculate summary statistics
    medians = [r['median_survival'] for r in km_results.values()]
    medians_finite = [m for m in medians if m is not None and not np.isinf(m)]
    
    # Use 90-day retention for best/worst comparison instead of median survival
    # This works even when median survival is "not reached"
    retention_90d_values = {
        cohort_col: results.get('retention_90d', 0) 
        for cohort_col, results in km_results.items()
    }
    
    if retention_90d_values:
        # Find best cohort (highest 90-day retention)
        best_cohort_col = max(retention_90d_values.items(), key=lambda x: x[1] if x[1] is not None else 0)
        best_cohort = (best_cohort_col[0], km_results[best_cohort_col[0]])
        best_retention = retention_90d_values[best_cohort_col[0]] * 100 if retention_90d_values[best_cohort_col[0]] else 0
        
        # Find worst cohort (lowest 90-day retention)
        worst_cohort_col = min(retention_90d_values.items(), key=lambda x: x[1] if x[1] is not None else 1)
        worst_cohort = (worst_cohort_col[0], km_results[worst_cohort_col[0]])
        worst_retention = retention_90d_values[worst_cohort_col[0]] * 100 if retention_90d_values[worst_cohort_col[0]] else 0
    else:
        best_cohort = (None, {'cohort_name': 'N/A'})
        worst_cohort = (None, {'cohort_name': 'N/A'})
        best_retention = 0
        worst_retention = 0
    
    if medians_finite:
        avg_median = np.mean(medians_finite)
    else:
        avg_median = 0
    
    # Row 1: Summary Metrics
    st.subheader("📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Cohorts Analyzed",
            value=len(km_results),
            help="Number of cohorts included in this analysis",
            border=True
        )
    
    with col2:
        if medians_finite:
            st.metric(
                label="Avg Median Survival",
                value=f"{avg_median:.0f} days",
                help="Average median survival time across all cohorts",
                border=True
            )
        else:
            st.metric(
                label="Avg Median Survival",
                value="Not reached",
                help="Most cohorts have >50% retention",
                border=True
            )
    
    with col3:
        st.metric(
            label="Best Cohort (90d)",
            value=best_cohort[1]['cohort_name'],
            delta=f"{best_retention:.1f}%",
            delta_color="normal",  # This removes the arrow!
            help="Cohort with highest 90-day retention rate",
            border=True
        )
    
    with col4:
        st.metric(
            label="Worst Cohort (90d)",
            value=worst_cohort[1]['cohort_name'],
            delta=f"{worst_retention:.1f}%",
            delta_color="inverse",  # This removes the arrow!
            help="Cohort with lowest 90-day retention rate",
            border=True
        )

    st.markdown("---")
    
    # Row 2: Kaplan-Meier Survival Curves
    st.subheader("📉 Survival Curves")
    
    fig_survival = plot_survival_curves(km_results)
    st.plotly_chart(fig_survival, use_container_width=True)
    
    st.info("""
    📊 **Reading the Chart:**
    - **Y-axis**: Probability of still being active (0% = all churned, 100% = all active)
    - **X-axis**: Days since user signup
    - **Shaded areas**: 95% confidence intervals
    - **Hover**: View exact values at any point in time
    """)
    
    st.markdown("---")
    
    # Row 3: Detailed Comparison Table
    st.subheader("📊 Cohort Comparison Table")
    
    summary_table = get_survival_summary_table(km_results)
    
    # Style the dataframe
    st.dataframe(
        summary_table,
        use_container_width=True,
        height=min(400, (len(summary_table) + 1) * 35)
    )
    
    st.caption("* 'Not reached' = Median survival undefined (>50% of cohort still active)")
    
    st.markdown("---")
    
    # Row 4: Statistical Comparisons
    st.subheader("🔬 Statistical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Hazard Ratios", "🔢 Pairwise Comparisons", "📋 Detailed Tests"])
    
    with tab1:
        st.markdown("### Cox Proportional Hazards Analysis")
        st.markdown("Hazard ratio > 1 indicates **higher churn risk** compared to reference cohort.")
        
        with st.spinner("Calculating hazard ratios..."):
            hazard_df = calculate_hazard_ratios(df, selected_cohort_cols)
        
        if not hazard_df.empty and 'error' not in hazard_df.columns:
            # Display as formatted table
            st.dataframe(
                hazard_df.style.format({
                    'hazard_ratio': '{:.2f}',
                    'ci_lower': '{:.2f}',
                    'ci_upper': '{:.2f}',
                    'p_value': '{:.4f}'
                }).background_gradient(subset=['hazard_ratio'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            # Visualize hazard ratios
            fig_hr = go.Figure()
            
            for idx, row in hazard_df.iterrows():
                if not pd.isna(row['p_value']):
                    color = '#e74c3c' if row['significant'] else '#95a5a6'
                    
                    fig_hr.add_trace(go.Scatter(
                        x=[row['hazard_ratio']],
                        y=[row['cohort_name']],
                        mode='markers',
                        marker=dict(size=12, color=color),
                        name=row['cohort_name'],
                        showlegend=False,
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[row['ci_upper'] - row['hazard_ratio']],
                            arrayminus=[row['hazard_ratio'] - row['ci_lower']],
                            color=color
                        )
                    ))
            
            # Add reference line at HR = 1
            fig_hr.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                           annotation_text="Reference (HR=1.0)")
            
            fig_hr.update_layout(
                title="Hazard Ratios with 95% Confidence Intervals",
                xaxis_title="Hazard Ratio",
                yaxis_title="",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_hr, use_container_width=True)
        else:
            st.warning("⚠️ " + hazard_df['error'].iloc[0] if 'error' in hazard_df.columns 
                      else "Unable to calculate hazard ratios.")
    
    with tab2:
        st.markdown("### Pairwise Log-Rank Test Results")
        st.markdown("Green = not significant (p > 0.05) | Red = significantly different (p < 0.05)")
        
        if len(selected_cohort_cols) >= 2:
            with st.spinner("Running pairwise comparisons..."):
                comparison_matrix = create_comparison_matrix(df, selected_cohort_cols)
            
            fig_heatmap = plot_comparison_heatmap(comparison_matrix)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("ℹ️ Select at least 2 cohorts to see pairwise comparisons.")
    
    with tab3:
        st.markdown("### Detailed Pairwise Comparisons")
        
        if len(selected_cohort_cols) >= 2:
            cohort_a = st.selectbox("Select first cohort:", selected_cohort_names, index=0)
            cohort_b = st.selectbox("Select second cohort:", selected_cohort_names, index=min(1, len(selected_cohort_names)-1))
            
            if cohort_a != cohort_b:
                result = compare_cohorts_logrank(
                    df,
                    cohort_mapping[cohort_a],
                    cohort_mapping[cohort_b]
                )
                
                if 'error' not in result:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Cohort A", cohort_a)
                        st.metric("Sample Size", f"{result['cohort_a_sample_size']:,}")
                        median_a = result['cohort_a_median']
                        st.metric("Median Survival", 
                                 f"{median_a:.0f} days" if median_a and not np.isinf(median_a) else "Not reached")
                    
                    with col_b:
                        st.metric("Cohort B", cohort_b)
                        st.metric("Sample Size", f"{result['cohort_b_sample_size']:,}")
                        median_b = result['cohort_b_median']
                        st.metric("Median Survival", 
                                 f"{median_b:.0f} days" if median_b and not np.isinf(median_b) else "Not reached")
                    
                    st.markdown("---")
                    st.markdown("#### Test Results")
                    st.write(f"**Test Statistic:** {result['test_statistic']:.4f}")
                    st.write(f"**P-value:** {result['p_value']:.6f}")
                    
                    if result['significant']:
                        st.success(f"✅ **{result['interpretation']}**")
                        st.info(f"📌 {result['survival_difference']}")
                    else:
                        st.info(f"ℹ️ **{result['interpretation']}**")
                else:
                    st.error(result['error'])
            else:
                st.warning("⚠️ Please select two different cohorts.")
        else:
            st.info("ℹ️ Select at least 2 cohorts for detailed comparisons.")
    
    st.markdown("---")
    
    # Row 5: Export Section
    st.subheader("💾 Export Analysis Results")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        # Export summary table
        csv_summary = summary_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary Table (CSV)",
            data=csv_summary,
            file_name=f"cohort_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export2:
        # Export survival data
        if not hazard_df.empty and 'error' not in hazard_df.columns:
            csv_hazard = hazard_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hazard Ratios (CSV)",
                data=csv_hazard,
                file_name=f"hazard_ratios_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


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
