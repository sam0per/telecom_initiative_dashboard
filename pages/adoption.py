import streamlit as st
import plotly.express as px
import pandas as pd
from data import get_adoption_data # Import data function

def display_metrics(df: pd.DataFrame):
    """Displays the key metrics for adoption and engagement."""
    st.subheader("Current Adoption & Engagement Levels")

    # Get latest data
    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2] # Compare with the previous day for delta

    metrics = {
        "Adoption Rate (%)": {"suffix": "%", "help": "Percentage of the target user base actively using the initiative/feature."},
        "DAU": {"suffix": "", "help": "Daily Active Users: Number of unique users engaging on a given day."},
        "MAU": {"suffix": "", "help": "Monthly Active Users: Number of unique users engaging over the last 30 days."},
        "DAU/MAU Ratio (%)": {"suffix": "%", "help": "Ratio of Daily to Monthly Active Users, indicating engagement intensity or 'stickiness'."}
    }

    cols = st.columns(len(metrics))

    for i, (metric, config) in enumerate(metrics.items()):
        value = latest_data[metric]
        delta = value - previous_data[metric]
        delta_off = None

        if abs(delta) > 1e-6:
            delta_off = f"{delta:,.0f}" if config['suffix'] == "" else f"{delta:.2f}{config['suffix']}"

        cols[i].metric(
            label=metric,
            value=f"{value:,.0f}" if config['suffix'] == "" else f"{value:.2f}{config['suffix']}",
            delta=delta_off,
            delta_color="normal", # Default delta color
            help=config["help"]
        )


def plot_trends(df: pd.DataFrame):
    """Plots the time series trends for adoption and engagement KPIs."""
    st.subheader("Adoption & Engagement Trends")

    # --- Adoption Rate Plot ---
    st.markdown("##### Adoption Rate")
    fig_adoption = px.line(
        df,
        y="Adoption Rate (%)",
        title="Initiative Adoption Rate Over Time",
        labels={"value": "Adoption Rate (%)", "Date": "Date"},
        template="plotly_white"
    )
    fig_adoption.update_layout(hovermode="x unified")
    st.plotly_chart(fig_adoption, use_container_width=True)
    st.markdown("*(Percentage of target users adopting the feature/initiative)*")
    st.divider()

    # --- DAU/MAU Plot ---
    st.markdown("##### Active Users (DAU & MAU)")
    fig_users = px.line(
        df,
        y=['DAU', 'MAU'],
        title="Daily vs Monthly Active Users",
        labels={"value": "Number of Users", "Date": "Date", "variable": "Metric"},
        template="plotly_white"
    )
    fig_users.update_layout(hovermode="x unified")
    st.plotly_chart(fig_users, use_container_width=True)
    st.markdown("*(Tracking daily and monthly unique user engagement)*")
    st.divider()

    # --- DAU/MAU Ratio (Engagement) Plot ---
    st.markdown("##### Engagement Intensity (DAU/MAU Ratio)")
    fig_ratio = px.line(
        df,
        y="DAU/MAU Ratio (%)",
        title="User Engagement Intensity (Stickiness)",
        labels={"value": "DAU/MAU Ratio (%)", "Date": "Date"},
        template="plotly_white"
    )
    fig_ratio.update_layout(hovermode="x unified")
    fig_ratio.update_traces(line_color='purple')
    st.plotly_chart(fig_ratio, use_container_width=True)
    st.markdown("*(Higher ratio indicates users engage more frequently)*")


def display_page():
    """Main function to display the Adoption & Engagement page."""
    st.title("📊 Initiative Adoption & Engagement")
    st.markdown("""
        Track how users are adopting the new initiative and monitor their engagement levels.
        These metrics help understand user behavior and feature stickiness.
    """)

    # Load data
    adoption_df = get_adoption_data()

    if adoption_df is not None and not adoption_df.empty:
        # Display Metrics
        display_metrics(adoption_df)

        st.markdown("---") # Visual separator

        # Display Trend Plots
        plot_trends(adoption_df)

        # Optional: Show raw data in an expander
        with st.expander("View Raw Adoption & Engagement Data"):
            st.dataframe(adoption_df)
    else:
        st.warning("Could not load adoption and engagement data.")

# This ensures the code runs only when the file is executed directly
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Adoption Demo")
    display_page()