import streamlit as st
import plotly.express as px
import pandas as pd
from data import get_overview_data # Import data function

def display_metrics(df: pd.DataFrame):
    """Displays the key metrics using st.metric."""
    st.subheader("Current Snapshot")

    # Get latest data and previous period data for delta calculation
    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2] # Compare with the previous day

    # Define metrics to display
    metrics = {
        "Revenue Impact (M$)": {"suffix": "M $", "help": "Estimated additional revenue generated by the initiative."},
        "Churn Rate (%)": {"suffix": "%", "delta_inverted": True, "help": "Percentage of customers lost. Lower is better."},
        "NPS": {"suffix": "", "help": "Net Promoter Score: Index ranging from -100 to 100 measuring customer willingness to recommend."}
    }

    cols = st.columns(len(metrics)) # Create columns for layout

    for i, (metric, config) in enumerate(metrics.items()):
        value = latest_data[metric]
        delta = value - previous_data[metric]
        delta_off = None # Default: Don't show delta if it's zero or negligible

        # Check if delta is significant enough to display
        if abs(delta) > 1e-6: # Avoid showing delta for tiny floating point differences
             delta_off = f"{delta:.2f}{config['suffix']}"

        cols[i].metric(
            label=metric,
            value=f"{value:.2f}{config['suffix']}" if config['suffix'] else f"{value:.0f}",
            delta=delta_off,
            delta_color="inverse" if config.get("delta_inverted", False) else "normal",
            help=config["help"]
        )

def plot_trends(df: pd.DataFrame):
    """Plots the time series trends for the overview KPIs."""
    st.subheader("KPI Trends Over Time")

    # --- Revenue Impact Plot ---
    st.markdown("##### Revenue Impact")
    fig_revenue = px.line(
        df,
        y="Revenue Impact (M$)",
        title="Revenue Impact Trend",
        labels={"value": "Revenue Impact (M$)", "Date": "Date"},
        template="plotly_white"
    )
    fig_revenue.update_layout(hovermode="x unified")
    st.plotly_chart(fig_revenue, use_container_width=True)
    st.markdown("*(Estimated additional revenue in millions of dollars)*")
    st.divider()

    # --- Churn Rate Plot ---
    st.markdown("##### Churn Rate")
    fig_churn = px.line(
        df,
        y="Churn Rate (%)",
        title="Churn Rate Trend",
        labels={"value": "Churn Rate (%)", "Date": "Date"},
        template="plotly_white"
    )
    fig_churn.update_layout(hovermode="x unified")
    fig_churn.update_traces(line_color='orange')
    st.plotly_chart(fig_churn, use_container_width=True)
    st.markdown("*(Percentage of subscribers who discontinue their service)*")
    st.divider()

    # --- NPS Plot ---
    st.markdown("##### Net Promoter Score (NPS)")
    fig_nps = px.line(
        df,
        y="NPS",
        title="NPS Trend",
        labels={"value": "NPS Score", "Date": "Date"},
        template="plotly_white"
    )
    fig_nps.update_layout(hovermode="x unified")
    fig_nps.update_traces(line_color='green')
    st.plotly_chart(fig_nps, use_container_width=True)
    st.markdown("*(Customer loyalty and satisfaction metric)*")


def display_page():
    """Main function to display the Overview page."""
    st.title("🚀 Initiative Performance Overview")
    st.markdown("""
        This page provides a high-level summary of the most critical Key Performance Indicators (KPIs)
        measuring the overall success and impact of our key initiatives.
    """)

    # Load data
    overview_df = get_overview_data()

    if overview_df is not None and not overview_df.empty:
        # Display Metrics
        display_metrics(overview_df)

        st.markdown("---") # Visual separator

        # Display Trend Plots
        plot_trends(overview_df)

        # Optional: Show raw data in an expander
        with st.expander("View Raw Overview Data"):
            st.dataframe(overview_df)
    else:
        st.warning("Could not load overview data.")

# This ensures the code runs only when the file is executed directly
# (useful for testing, though not strictly necessary for Streamlit pages)
if __name__ == "__main__":
    # You could add test code here if needed
    st.set_page_config(layout="wide", page_title="Overview Demo")
    display_page()