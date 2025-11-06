import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
N_DAYS = 90  # Number of days for time series data
START_DATE = datetime.now() - timedelta(days=N_DAYS)

# --- Helper Functions ---
def _generate_time_series_data(start_date, n_days):
    """Generates a base DataFrame with a date range."""
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    return pd.DataFrame({'Date': dates})

def _add_trend(series, initial_value, trend_factor, noise_level=0.1):
    """Adds a trend and noise to a series."""
    days = np.arange(len(series))
    trend = initial_value + days * trend_factor
    noise = np.random.normal(0, noise_level * initial_value, len(series))
    return np.maximum(0, trend + noise) # Ensure non-negative values where applicable

# --- Data Generation Functions ---

@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_overview_data():
    """Generates mock data for the Overview page KPIs."""
    df = _generate_time_series_data(START_DATE, N_DAYS)

    # Revenue Impact ($ Million) - positive trend
    df['Revenue Impact (M$)'] = _add_trend(df['Date'], 5.0, 0.05, noise_level=0.15)

    # Churn Reduction (%) - positive trend (meaning churn rate decreases)
    initial_churn = 2.5 # Starting churn rate %
    churn_reduction_trend = -0.01 # Churn rate decreases over time
    df['Churn Rate (%)'] = _add_trend(df['Date'], initial_churn, churn_reduction_trend, noise_level=0.1)
    df['Churn Rate (%)'] = df['Churn Rate (%)'].clip(lower=0.5) # Don't let churn go below 0.5%

    # NPS (Net Promoter Score) - positive trend
    df['NPS'] = _add_trend(df['Date'], 30, 0.2, noise_level=0.08).astype(int)
    df['NPS'] = df['NPS'].clip(lower=-100, upper=100) # NPS range

    return df.set_index('Date')

@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_adoption_data():
    """Generates mock data for the Adoption & Engagement page KPIs."""
    df = _generate_time_series_data(START_DATE, N_DAYS)

    # Adoption Rate (%) - S-curve like growth
    days = np.arange(N_DAYS)
    # Sigmoid function for S-curve: L / (1 + exp(-k*(x-x0)))
    L = 80 # Max adoption %
    k = 0.1 # Steepness
    x0 = N_DAYS / 2 # Midpoint
    sigmoid = L / (1 + np.exp(-k * (days - x0)))
    noise = np.random.normal(0, 2, N_DAYS)
    sig_values = np.maximum(0, sigmoid + noise)  # Ensure values are >= 0
    df["Adoption Rate (%)"] = np.minimum(sig_values, 100)  # Ensure values are <= 100

    # MAU (Monthly Active Users) - general upward trend
    df['MAU'] = _add_trend(df['Date'], 50000, 500, noise_level=0.05).astype(int)

    # DAU (Daily Active Users) - more volatile than MAU, related trend
    # Let DAU be roughly 10-20% of MAU with daily fluctuations
    dau_ratio = np.random.uniform(0.10, 0.20, N_DAYS)
    df['DAU'] = (df['MAU'] * dau_ratio).astype(int)
    df['DAU'] = df['DAU'] + np.random.normal(0, df['DAU'] * 0.05, N_DAYS).astype(int) # Add some noise
    df['DAU'] = df['DAU'].clip(lower=1000) # Ensure minimum DAU

    # Calculate DAU/MAU Ratio (%) for engagement trend
    # Use rolling average for MAU to make ratio smoother and more realistic
    # Note: A simple daily MAU is used here for simplicity, a real calc would be more complex
    df['DAU/MAU Ratio (%)'] = (df['DAU'] / df['MAU']) * 100

    return df.set_index('Date')


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cohort_data():
    """
    Generates mock user-level data for cohort survival analysis.
    
    Returns:
        DataFrame with columns:
        - user_id: Unique identifier
        - signup_date: User registration date
        - last_activity_date: Last recorded activity
        - is_churned: Boolean churn status
        - churn_date: Date of churn (if churned)
        - user_segment: Enterprise/SMB/Consumer
        - region: North/South/East/West
        - plan_type: Premium/Standard/Basic
        - signup_quarter: Q1/Q2/Q3/Q4
        - days_active: Total days from signup to churn/censoring
        - cohort_month: YYYY-MM format
    """
    np.random.seed(42)  # Reproducibility
    
    n_users = 10000
    
    # Generate signup dates over past 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    signup_dates = pd.to_datetime([
        start_date + timedelta(days=np.random.randint(0, 365))
        for _ in range(n_users)
    ])
    
    # User segments with defined proportions
    segments = np.random.choice(
        ['Enterprise', 'SMB', 'Consumer'],
        size=n_users,
        p=[0.20, 0.30, 0.50]
    )
    
    # Regions with equal distribution
    regions = np.random.choice(
        ['North', 'South', 'East', 'West'],
        size=n_users,
        p=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Plan types
    plan_types = np.random.choice(
        ['Premium', 'Standard', 'Basic'],
        size=n_users,
        p=[0.30, 0.50, 0.20]
    )
    
    # Calculate signup quarters
    signup_quarters = ['Q' + str((month - 1) // 3 + 1) 
                      for month in pd.DatetimeIndex(signup_dates).month]
    
    # Generate churn based on segment-specific survival curves
    # Using segment-specific parameters for realistic patterns
    segment_params = {
        'Enterprise': {'base_churn_rate': 0.18, 'mean_lifetime': 300, 'shape': 1.5},
        'SMB': {'base_churn_rate': 0.35, 'mean_lifetime': 180, 'shape': 1.2},
        'Consumer': {'base_churn_rate': 0.45, 'mean_lifetime': 120, 'shape': 1.0}
    }
    
    days_active_list = []
    is_churned_list = []
    churn_dates_list = []
    
    for i in range(n_users):
        segment = segments[i]
        signup_date = signup_dates[i]
        quarter = signup_quarters[i]
        region = regions[i]
        
        params = segment_params[segment]
        base_churn = params['base_churn_rate']
        
        # Apply modifiers
        # Q4 signups have better retention (10% boost)
        churn_modifier = 0.9 if quarter == 'Q4' else 1.0
        
        # Regional variation (±5%)
        regional_modifier = {
            'North': 0.95,
            'South': 1.05,
            'East': 1.0,
            'West': 1.0
        }[region]
        
        adjusted_churn_rate = base_churn * churn_modifier * regional_modifier
        
        # Determine if user churned
        is_churned = np.random.random() < adjusted_churn_rate
        
        # Generate days active using Weibull distribution for realistic survival curves
        # Scale based on mean lifetime, shape controls hazard rate pattern
        days_active = int(np.random.weibull(params['shape']) * params['mean_lifetime'])
        
        # Ensure days_active doesn't exceed time since signup
        max_days_possible = (end_date - signup_date).days
        days_active = min(days_active, max_days_possible)
        
        # For non-churned users, set days_active to time since signup
        if not is_churned:
            days_active = max_days_possible
            churn_date = pd.NaT
        else:
            churn_date = signup_date + timedelta(days=days_active)
        
        days_active_list.append(days_active)
        is_churned_list.append(is_churned)
        churn_dates_list.append(churn_date)
    
    # Calculate last activity date
    last_activity_dates = [
        signup_dates[i] + timedelta(days=days_active_list[i])
        for i in range(n_users)
    ]
    
    # Create cohort month (YYYY-MM)
    cohort_months = [date.strftime('%Y-%m') for date in signup_dates]
    
    # Build DataFrame
    df = pd.DataFrame({
        'user_id': [f'USR-{str(i).zfill(6)}' for i in range(1, n_users + 1)],
        'signup_date': signup_dates,
        'last_activity_date': last_activity_dates,
        'is_churned': is_churned_list,
        'churn_date': churn_dates_list,
        'user_segment': segments,
        'region': regions,
        'plan_type': plan_types,
        'signup_quarter': signup_quarters,
        'days_active': days_active_list,
        'cohort_month': cohort_months
    })
    
    return df


# Example usage (optional, for testing data.py directly)
if __name__ == '__main__':
    overview_data = get_overview_data()
    print("Overview Data Sample:\n", overview_data.tail())

    adoption_data = get_adoption_data()
    print("\nAdoption Data Sample:\n", adoption_data.tail())
    
    # Test new cohort data
    cohort_data = get_cohort_data()
    print("\nCohort Data Sample:\n", cohort_data.head(10))
    print("\nCohort Data Info:")
    print(f"Total users: {len(cohort_data)}")
    print(f"Churned users: {cohort_data['is_churned'].sum()} ({cohort_data['is_churned'].mean():.1%})")
    print("\nChurn by segment:")
    print(cohort_data.groupby('user_segment')['is_churned'].agg(['sum', 'mean']))
