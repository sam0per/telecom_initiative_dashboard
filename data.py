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
    df['Adoption Rate (%)'] = np.maximum(0, sigmoid + noise).clip(upper=100)

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

# Example usage (optional, for testing data.py directly)
if __name__ == '__main__':
    overview_data = get_overview_data()
    print("Overview Data Sample:\n", overview_data.tail())

    adoption_data = get_adoption_data()
    print("\nAdoption Data Sample:\n", adoption_data.tail())