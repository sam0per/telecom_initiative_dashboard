import streamlit as st
from module import overview, adoption # Import page modules

# --- Page Configuration ---
st.set_page_config(
    page_title="Telecom Initiative Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Page Definitions ---
# A dictionary mapping page names to their display functions
PAGES = {
    "🏠 Overview": overview.display_page,
    "📈 Adoption & Engagement": adoption.display_page,
    # Add new pages here:
    # "💡 New KPI Page": new_kpi_page.display_page,
}

# --- Sidebar Navigation ---
st.sidebar.title("🌐 Navigation")
st.sidebar.markdown("Select a page to view initiative KPIs:")

# Create radio buttons for page selection
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Add a drop-down list for initiative selection
initiative = st.sidebar.selectbox(
    "🎯 Select an Initiative:",
    ["All Initiatives", "Initiative 1", "Initiative 2", "Initiative 3"],
    index=0,  # Default to "All Initiatives"
)

# --- Page Routing ---
# Get the function corresponding to the selected page
page_function = PAGES[selection]

# Call the selected page's display function
page_function()

# --- Optional: Footer or common elements ---
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **About:** This dashboard tracks the success of key telecom initiatives.
    Contact [Your Team/Email] for support.
    """
)
