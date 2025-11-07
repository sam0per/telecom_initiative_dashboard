# 📊 Telecom Initiative Performance Dashboard

A comprehensive, interactive analytics dashboard built with **Streamlit** for tracking and analyzing telecom initiative performance. Features advanced cohort analysis, survival modeling, and data visualization capabilities.
<img width="936" height="524" alt="DataPipelineVisualization" src="https://github.com/user-attachments/assets/0c0ea9b8-64ad-4433-a813-cbc1def9fb85" />

---

## ✨ Features

### 📈 **Overview Dashboard**
- Real-time KPI tracking (Revenue Impact, Churn Rate, NPS)
- Interactive time-series visualizations with Plotly
- Performance metrics with delta indicators
- Historical trend analysis

### 📊 **Adoption & Engagement Analytics**
- User adoption rate tracking
- Daily Active Users (DAU) and Monthly Active Users (MAU) monitoring
- Engagement intensity metrics (DAU/MAU ratio)
- Growth trend visualizations

### 🧬 **Cohort Survival Analysis** *(New)*
- **Kaplan-Meier survival curves** for retention analysis
- Multi-cohort comparison with statistical testing
- **Hazard ratio analysis** using Cox Proportional Hazards model
- Time-to-event analysis (churn prediction)
- Comprehensive cohort segmentation:
  - User segments (Enterprise, SMB, Consumer)
  - Regional cohorts (North, South, East, West)
  - Plan types (Premium, Standard, Basic)
  - Seasonal cohorts (Quarterly signups)
- Interactive visualizations with confidence intervals
- Statistical comparison tools (log-rank tests, pairwise analysis)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (recommended: 3.11 or 3.12)
- **uv** package manager (recommended) or pip

### Installation

#### Option 1: Using `uv` (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/sam0per/telecom_initiative_dashboard.git
cd telecom_initiative_dashboard

# Install dependencies
uv sync
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/sam0per/telecom_initiative_dashboard.git
cd telecom_initiative_dashboard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# With uv
uv run streamlit run app.py

# With pip
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## 📂 Project Structure

```
telecom_initiative_dashboard/
├── app.py                          # Main Streamlit application
├── data.py                         # Mock data generation functions
├── requirements.txt                # pip dependencies
├── pyproject.toml                  # uv/project configuration
│
├── module/                         # Page modules
│   ├── __init__.py
│   ├── overview.py                 # Overview dashboard page
│   ├── adoption.py                 # Adoption & engagement page
│   └── cohort_analysis.py          # Cohort survival analysis page
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   └── cohort_builder.py           # Cohort definition & time-to-event calculations
│
├── doc/                            # Documentation
│   └── healthcare_features_doc.md  # Feature development guide
│
└── tests/                          # Unit tests (to be added)
```

---

## 📊 Dashboard Pages

### 1. 🏠 Overview
High-level summary of critical KPIs:
- **Revenue Impact**: Estimated additional revenue (M$)
- **Churn Rate**: Customer retention metrics (%)
- **Net Promoter Score (NPS)**: Customer satisfaction index

**Features:**
- Current snapshot with period-over-period deltas
- Time-series trend charts
- Interactive hover details

### 2. 📈 Adoption & Engagement
User adoption and engagement tracking:
- **Adoption Rate**: % of target users actively using initiatives
- **DAU/MAU**: Daily and monthly active user counts
- **Engagement Ratio**: Stickiness indicator (DAU/MAU %)

**Features:**
- Multi-metric trend visualization
- Engagement intensity analysis
- S-curve adoption modeling

### 3. 🧬 Cohort Analysis
Advanced survival analysis for user retention:
- **Kaplan-Meier Curves**: Visual survival probability over time
- **Statistical Testing**: Log-rank tests for cohort comparison
- **Hazard Ratios**: Cox PH model for relative churn risk
- **Retention Metrics**: 30/60/90/180-day retention rates

**Features:**
- Multiple cohort selection (segments, regions, plans, seasons)
- Interactive survival curve plots with confidence intervals
- Pairwise cohort comparison matrix
- Detailed statistical analysis
- Export analysis results (CSV)

---

## 🔧 Technical Details

### Dependencies

**Core Libraries:**
- `streamlit` >= 1.20.0 - Web application framework
- `pandas` >= 1.4.0 - Data manipulation
- `plotly` >= 5.10.0 - Interactive visualizations
- `numpy` >= 1.20.0 - Numerical computing

**Analytics Libraries:**
- `lifelines` >= 0.30.0 - Survival analysis (Kaplan-Meier, Cox PH)
- `scipy` >= 1.16.3 - Statistical functions
- `scikit-learn` >= 1.7.2 - Machine learning utilities

**Data Generation:**
- `faker` >= 37.12.0 - Synthetic data generation

**Development Tools:**
- `pytest` >= 8.4.2 - Testing framework
- `black` >= 25.9.0 - Code formatting
- `flake8` >= 7.3.0 - Code linting

### Data Caching

The dashboard uses `@st.cache_data` with a 10-minute TTL (Time To Live) for optimal performance:
- Overview data: Cached for 600 seconds
- Adoption data: Cached for 600 seconds
- Cohort data: Cached for 600 seconds

Clear cache via Streamlit's UI menu: **☰ → Clear cache**

---

## 📈 Usage Examples

### Running Cohort Analysis

1. Navigate to **🧬 Cohort Analysis** page
2. Select cohorts from the sidebar (e.g., "User Segments")
3. View survival curves and retention metrics
4. Compare cohorts using statistical tests
5. Export results for further analysis

### Understanding Survival Curves

- **Y-axis**: Probability of remaining active (0-100%)
- **X-axis**: Days since user signup
- **Shaded areas**: 95% confidence intervals
- **Higher curves**: Better retention

### Interpreting Hazard Ratios

- **HR = 1.0**: Reference cohort (baseline risk)
- **HR > 1.0**: Higher churn risk than reference
- **HR < 1.0**: Lower churn risk (better retention)
- **Green bars**: Statistically significant differences (p < 0.05)

---

## 🧪 Testing the Utilities

### Cohort Builder

```bash
# Test cohort builder functions
uv run python utils/cohort_builder.py
```

Expected output:
- ✅ Loaded N users
- ✅ Created cohort definitions
- ✅ Assigned cohorts with boolean columns
- ✅ Calculated survival data (duration, event_observed)
- ✅ Generated cohort summary statistics

### Cohort Analysis

```bash
# Test Kaplan-Meier analysis
uv run python module/cohort_analysis.py
```

Expected output:
- ✅ Kaplan-Meier analysis for selected cohorts
- ✅ Survival summary table
- ✅ Log-rank test results
- ✅ Hazard ratios (Cox PH)
- ✅ Comparison matrix

---

## 🛠️ Development

### Adding a New Page

1. **Create module file**: `module/new_page.py`
2. **Implement display function**:
   ```python
   def display_page():
       st.title("New Page Title")
       # Your page logic here
   ```
3. **Register in app.py**:
   ```python
   from module import new_page
   
   PAGES = {
       # ...existing pages...
       "🆕 New Page": new_page.display_page,
   }
   ```

### Code Formatting

```bash
# Format code with black
uv run black .

# Lint with flake8
uv run flake8 .
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=utils --cov=module
```

---

## 📚 Key Metrics Glossary

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Adoption Rate** | % of target users actively using the initiative | Higher = better feature uptake |
| **DAU/MAU Ratio** | Daily active / Monthly active users | Higher = more engaged users |
| **NPS** | Net Promoter Score (-100 to +100) | > 0 is good, > 50 is excellent |
| **Churn Rate** | % of users who discontinue service | Lower = better retention |
| **Median Survival** | Time until 50% of cohort has churned | Higher = better retention |
| **Hazard Ratio** | Relative churn risk vs reference cohort | < 1 = lower risk, > 1 = higher risk |

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README for user-facing changes

---

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/sam0per/telecom_initiative_dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sam0per/telecom_initiative_dashboard/discussions)
- **Documentation**: See `doc/` folder for detailed feature guides

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🎯 Roadmap

### Planned Features
- [ ] **Performance Optimization Module**: Memory-efficient data processing benchmarks
- [ ] **Synthetic Healthcare Data Generator**: OMOP CDM-compliant patient data
- [ ] **Customer Satisfaction Tracking**: NPS drill-down analysis
- [ ] **Predictive Analytics**: ML-based churn prediction models
- [ ] **Export Functionality**: PDF/Excel report generation

### Recent Updates
- ✅ **v0.2.0**: Added Cohort Survival Analysis with Kaplan-Meier curves
- ✅ **v0.1.0**: Initial release with Overview and Adoption pages

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web apps
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Lifelines](https://lifelines.readthedocs.io/) - Survival analysis
- [Pandas](https://pandas.pydata.org/) - Data manipulation

---

**Made with ❤️ for data-driven decision making**
