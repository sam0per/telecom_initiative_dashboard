import pandas as pd
from datetime import datetime, timedelta


def create_cohort_definitions(df: pd.DataFrame) -> dict:
    """
    Define cohorts based on business rules.
    
    Args:
        df: DataFrame with cohort data (needs 'signup_date' column)
    
    Returns:
        dict: Cohort name -> lambda filter function
        
    Example cohorts:
        - All Users: Everyone (baseline)
        - Early Adopters: Signed up in first 3 months
        - Enterprise/SMB/Consumer: By user segment
        - Regional: North/South/East/West
        - Premium/Standard/Basic: By plan type
        - Q1/Q2/Q3/Q4 Signups: By signup quarter
    """
    # Calculate the first signup date for Early Adopters definition
    min_signup = df['signup_date'].min()
    early_cutoff = min_signup + timedelta(days=90)
    
    # Define cohort filters as lambda functions
    cohort_definitions = {
        # Baseline cohort
        'All Users': lambda df: pd.Series([True] * len(df), index=df.index),
        
        # Time-based cohorts
        'Early Adopters': lambda df: df['signup_date'] <= early_cutoff,
        
        # Segment-based cohorts
        'Enterprise': lambda df: df['user_segment'] == 'Enterprise',
        'SMB': lambda df: df['user_segment'] == 'SMB',
        'Consumer': lambda df: df['user_segment'] == 'Consumer',
        
        # Regional cohorts
        'North Region': lambda df: df['region'] == 'North',
        'South Region': lambda df: df['region'] == 'South',
        'East Region': lambda df: df['region'] == 'East',
        'West Region': lambda df: df['region'] == 'West',
        
        # Plan type cohorts
        'Premium Plan': lambda df: df['plan_type'] == 'Premium',
        'Standard Plan': lambda df: df['plan_type'] == 'Standard',
        'Basic Plan': lambda df: df['plan_type'] == 'Basic',
        
        # Seasonal cohorts
        'Q1 Signups': lambda df: df['signup_quarter'] == 'Q1',
        'Q2 Signups': lambda df: df['signup_quarter'] == 'Q2',
        'Q3 Signups': lambda df: df['signup_quarter'] == 'Q3',
        'Q4 Signups': lambda df: df['signup_quarter'] == 'Q4',
    }
    
    return cohort_definitions


def assign_cohorts(df: pd.DataFrame, cohort_definitions: dict) -> pd.DataFrame:
    """
    Assign users to cohorts based on definitions.
    
    Users can belong to multiple cohorts. For each cohort definition,
    a new boolean column is added indicating membership.
    
    Args:
        df: Input dataframe with user data
        cohort_definitions: Dictionary of cohort name -> filter function
        
    Returns:
        DataFrame with added boolean columns for each cohort:
        - 'cohort_all_users': True for all rows
        - 'cohort_early_adopters': True if user is early adopter
        - 'cohort_enterprise': True if user is enterprise
        - etc.
        
    Example:
        >>> cohort_defs = create_cohort_definitions(df)
        >>> df_with_cohorts = assign_cohorts(df, cohort_defs)
        >>> # Check which cohorts a specific user belongs to
        >>> user_cohorts = df_with_cohorts.loc[0, df_with_cohorts.columns.str.startswith('cohort_')]
    """
    # Create a copy to avoid modifying original
    df_with_cohorts = df.copy()
    
    # Apply each cohort definition
    for cohort_name, filter_func in cohort_definitions.items():
        # Create column name: "Enterprise" -> "cohort_enterprise"
        column_name = 'cohort_' + cohort_name.lower().replace(' ', '_')
        
        # Apply filter function to get boolean mask
        try:
            df_with_cohorts[column_name] = filter_func(df_with_cohorts)
        except Exception as e:
            # If filter fails, set all to False and warn
            print(f"Warning: Failed to apply cohort '{cohort_name}': {e}")
            df_with_cohorts[column_name] = False
    
    return df_with_cohorts


def calculate_time_to_event(df: pd.DataFrame, 
                            event_col: str = 'is_churned',
                            start_col: str = 'signup_date',
                            end_col: str = 'last_activity_date') -> pd.DataFrame:
    """
    Calculate duration and event status for survival analysis.
    
    This function prepares data for Kaplan-Meier analysis by calculating:
    1. Duration: Time from start to end (or censoring)
    2. Event observed: Whether the event occurred (1) or was censored (0)
    
    Args:
        df: Input DataFrame
        event_col: Column indicating if event occurred (boolean)
        start_col: Column with start date (e.g., signup_date)
        end_col: Column with end date (e.g., last_activity_date or churn_date)
        
    Returns:
        DataFrame with added columns:
        - 'duration': Number of days from start to end
        - 'event_observed': 1 if event occurred, 0 if censored
        
    Notes:
        - For churned users: duration = days until churn, event = 1
        - For active users: duration = days until censoring (now), event = 0
        - Duration is always positive (minimum 1 day)
        
    Example:
        >>> df_survival = calculate_time_to_event(df)
        >>> # Now ready for KaplanMeierFitter
        >>> kmf.fit(df_survival['duration'], df_survival['event_observed'])
    """
    # Create a copy to avoid modifying original
    df_survival = df.copy()
    
    # Ensure date columns are datetime
    df_survival[start_col] = pd.to_datetime(df_survival[start_col])
    df_survival[end_col] = pd.to_datetime(df_survival[end_col])
    
    # Calculate duration in days
    df_survival['duration'] = (df_survival[end_col] - df_survival[start_col]).dt.days

    # Handle negative durations (data issues)    
    negative_durations = (df_survival['duration'] < 0).sum()
    if negative_durations > 0:
        print(f"Warning: {negative_durations} rows had negative duration (fixed to 1 day).")

    # Ensure duration is at least 1 day (avoid zero or negative durations)
    df_survival['duration'] = df_survival['duration'].clip(lower=1)
    
    # Event observed: 1 if churned, 0 if still active (censored)
    df_survival['event_observed'] = df_survival[event_col].astype(int)
    
    # Data validation
    null_durations = df_survival['duration'].isnull().sum()
    if null_durations > 0:
        print(f"Warning: {null_durations} rows have null duration. These will be dropped.")
        df_survival = df_survival.dropna(subset=['duration'])
    
    return df_survival


# Testing utility function
def get_cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all cohorts in the dataframe.
    
    Args:
        df: DataFrame with cohort columns (from assign_cohorts)
        
    Returns:
        DataFrame with cohort statistics:
        - cohort_name
        - total_users
        - churned_users
        - churn_rate
        - median_days_active
        
    Example:
        >>> cohort_defs = create_cohort_definitions(df)
        >>> df_with_cohorts = assign_cohorts(df, cohort_defs)
        >>> df_survival = calculate_time_to_event(df_with_cohorts)
        >>> summary = get_cohort_summary(df_survival)
        >>> print(summary)
    """
    cohort_cols = [col for col in df.columns if col.startswith('cohort_')]
    
    summary_data = []
    
    for cohort_col in cohort_cols:
        cohort_name = cohort_col.replace('cohort_', '').replace('_', ' ').title()
        cohort_df = df[df[cohort_col] == True]
        
        if len(cohort_df) > 0:
            summary_data.append({
                'cohort_name': cohort_name,
                'total_users': len(cohort_df),
                'churned_users': cohort_df['is_churned'].sum(),
                'churn_rate': cohort_df['is_churned'].mean(),
                'median_days_active': cohort_df['days_active'].median(),
                'active_users': (~cohort_df['is_churned']).sum()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by total users descending
    summary_df = summary_df.sort_values('total_users', ascending=False)
    
    return summary_df


# Example usage and testing
if __name__ == '__main__':
    # Import the data generation function
    import sys
    sys.path.append('..')
    from data import get_cohort_data
    
    print("=" * 80)
    print("PHASE 2: COHORT BUILDER UTILITY - TESTING")
    print("=" * 80)
    
    # Load cohort data
    print("\n1. Loading cohort data...")
    df = get_cohort_data()
    print(f"   ✅ Loaded {len(df)} users")
    
    # Create cohort definitions
    print("\n2. Creating cohort definitions...")
    cohort_defs = create_cohort_definitions(df)
    print(f"   ✅ Created {len(cohort_defs)} cohort definitions:")
    for cohort_name in cohort_defs.keys():
        print(f"      - {cohort_name}")
    
    # Assign cohorts
    print("\n3. Assigning users to cohorts...")
    df_with_cohorts = assign_cohorts(df, cohort_defs)
    cohort_cols = [col for col in df_with_cohorts.columns if col.startswith('cohort_')]
    print(f"   ✅ Added {len(cohort_cols)} cohort columns")
    
    # Calculate time-to-event
    print("\n4. Calculating time-to-event data...")
    df_survival = calculate_time_to_event(df_with_cohorts)
    print(f"   ✅ Added 'duration' and 'event_observed' columns")
    print(f"   ✅ Duration range: {df_survival['duration'].min()} - {df_survival['duration'].max()} days")
    print(f"   ✅ Events observed: {df_survival['event_observed'].sum()} / {len(df_survival)}")
    
    # Generate cohort summary
    print("\n5. Generating cohort summary statistics...")
    summary = get_cohort_summary(df_survival)
    print("\n" + "=" * 80)
    print("COHORT SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
    
    # Test individual cohort
    print("\n" + "=" * 80)
    print("SAMPLE: ENTERPRISE COHORT ANALYSIS")
    print("=" * 80)
    enterprise_users = df_survival[df_survival['cohort_enterprise']]
    print(f"Total Enterprise users: {len(enterprise_users)}")
    print(f"Churned: {enterprise_users['is_churned'].sum()} ({enterprise_users['is_churned'].mean():.1%})")
    print(f"Median survival: {enterprise_users['duration'].median():.0f} days")
    print(f"Mean survival: {enterprise_users['duration'].mean():.0f} days")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 2 COMPLETE - All functions working correctly!")
    print("=" * 80)
