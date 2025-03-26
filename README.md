### How to Run:
1. Save the files in the specified directory structure.
2. Open your terminal or command prompt.
3. Navigate to the telecom_initiative_dashboard directory.
4. Install the required libraries: pip install -r requirements.txt
5. Run the Streamlit app: streamlit run app.py

Your browser should automatically open a new tab displaying the dashboard. You can navigate between the "Overview" and "Adoption & Engagement" pages using the sidebar.

### Explanation of Best Practices Applied:
1. **Modularization:** Code is split into logical files (app.py, data.py, pages/overview.py, pages/adoption.py). This makes it easier to manage, update, and debug specific parts of the application.
2. **Scalability:** Adding a new KPI page is straightforward:
   1. Create a new Python file in the pages directory (e.g., customer_satisfaction.py).
   2. Implement the display_page() function within that file, including data loading (potentially adding a function to data.py), metrics, and plots.
   3. Import the new module in app.py.
   4. Add the new page entry to the PAGES dictionary in app.py. The sidebar navigation will update automatically.
3. **Readability & Maintainability:**
   1. Code is organized into functions (display_metrics, plot_trends, display_page).
   2. Meaningful variable names are used.
   3. Docstrings explain the purpose of functions.
   4. Comments clarify specific logic where needed.
   5. Pythonic practices (e.g., using f-strings, dictionary lookups) are employed.
4. **User-Friendliness (Non-Technical Users):**
   1. Clear titles and markdown descriptions explain the purpose of each page and metric.
   2. st.metric provides clear, concise KPI snapshots with optional deltas for context. Tooltips (help parameter) explain metric definitions.
   3. Interactive Plotly charts allow users to hover, zoom, and explore data trends visually. Chart titles and axis labels are clear.
   4. Simple sidebar navigation (st.sidebar.radio) is intuitive.
5. **Performance:**
   1. @st.cache_data is used on data loading functions (get_overview_data, get_adoption_data) in data.py. This prevents regenerating or reloading mock data on every interaction, significantly speeding up the app. The ttl argument ensures data refreshes periodically (e.g., every 10 minutes).
6. **Data Separation:** Mock data generation is isolated in data.py. This makes it easy to replace the mock data functions with connections to real databases or APIs later without changing the page logic significantly.
7. **Visualization:** Plotly Express (px) is used for creating visually appealing and interactive charts with minimal code. Layout options (hovermode="x unified") enhance usability.