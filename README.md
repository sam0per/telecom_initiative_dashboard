# 📊 Telecom Initiative Dashboard

Welcome to the **Telecom Initiative Dashboard**! This interactive dashboard helps you track and analyze key performance indicators (KPIs) for various telecom initiatives. Built with **Streamlit**, it provides a user-friendly interface for exploring data trends and insights.

---

## 🚀 Features
- **📈 Interactive Charts**: Explore trends with Plotly-powered visualizations.
- **📊 KPI Snapshots**: Get quick insights with metrics and deltas.
- **🌐 Easy Navigation**: Switch between pages and initiatives effortlessly.
- **⚡ Performance Optimized**: Cached data ensures a smooth experience.

---

## 🎯 How to Run

Follow these steps to get started:

1. **Clone the Repository**:
```bash
git clone https://github.com/sam0per/telecom_initiative_dashboard.git
cd telecom_initiative_dashboard
```
2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
3. **Run the App**:
```bash
streamlit run app.py
```
4. **Open in Browser**: Your browser will automatically open the dashboard. If not, navigate to:
```bash
http://localhost:8501
```

---

## 🖼️ Preview

### **Dashboard Overview**
![Dashboard Overview](images/dashboard_overview_screenshot.png)

---

## 🛠️ Best Practices Applied

**1. Modularization**  
Code is split into logical files:
- `app.py`: Main entry point for the app.
- `data.py`: Handles mock data generation.
- `module/overview.py` & `module/adoption.py`: Define page-specific logic.

**2. Scalability**  
Adding a new KPI page is simple:
- Create a new Python file in the `module` directory (e.g., `module/customer_satisfaction.py`).
- Implement the `display_page()` function.
- Import the new module in `app.py` and add it to the `PAGES` dictionary.

**3. User-Friendliness**  
- **Clear Titles**: Each page has a descriptive title.
- **Tooltips**: Metrics include helpful explanations.
- **Interactive Charts**: Hover, zoom, and explore data visually.

**4. Performance**  
- **Caching**: `@st.cache_data` ensures data is not regenerated unnecessarily.
- **Efficient Layout**: Wide layout maximizes screen space.

---

## 📊 Example KPIs

| **Metric**            | **Description**                                   |
|-----------------------|---------------------------------------------------|
| **Adoption Rate (%)** | Percentage of users adopting the initiative.      |
| **DAU/MAU Ratio (%)** | Engagement intensity of users.                    |
| **NPS (Score)**       | Net Promoter Score for customer satisfaction.     |

---

## 🤝 Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```
3. Commit your changes and push:
```bash
git push origin feature/your-feature-name
```
4. Open a pull request.