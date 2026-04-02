# azure_demand_optimization

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-LICENSE-green)

## 📝 Description

azure_demand_optimization is a comprehensive, end-to-end Python framework designed for precise Azure demand forecasting and capacity optimization. It streamlines the entire lifecycle of cloud resource planning by automating data preprocessing, implementing robust missing-value treatment, and ensuring data consistency through canonical region mapping. By integrating advanced time-series usage analysis with intuitive, per-region visualization dashboards, this solution provides cloud administrators and data scientists with the actionable insights required to minimize over-provisioning and optimize resource allocation across global Azure environments.

## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
streamlit: 1.35.0
pandas: 2.2.2
numpy: 1.26.4
plotly: 5.22.0
scikit-learn: 1.5.0
xgboost: 2.0.3
statsmodels: 0.14.2
joblib: 1.4.2
schedule: 1.2.2
fastapi: 0.111.0
uvicorn: 0.30.1
pydantic: 2.7.4
pillow: 10.4.0
```

## 📁 Project Structure

```
.
├── Azure_Based_Demand_Forecasting_Data.csv
├── LICENSE
├── actual_vs_predicted.png
├── api.py
├── api_log.txt
├── azure_demand.ipynb
├── azure_demand.py
├── batch_log.txt
├── batch_predict.py
├── best_arima_model.pkl
├── best_xgboost_model.pkl
├── dashboard_old_app.py
├── demand_forecast_comparison.png
├── feature_importance.png
├── forecast_output.csv
├── gitignore.txt
├── milestone_3 (1).py
├── milestone_two.py
├── model_rmse_comparison.png
├── monitoring.py
├── monitoring_log.txt
├── new_data.csv
├── requirements.txt
├── rmse_history.csv
├── runtime.txt
├── scheduler.py
├── scheduler_log.txt
├── usage_units_Central-India.png
├── usage_units_East-US.png
├── usage_units_southeast-asia.png
└── usage_units_west-europe.png
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: https://github.com/s22wakadesneha-dev/azure_demand_optimization.git
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request

Please ensure your code follows the project's style guidelines and includes tests where applicable.

## 📜 License

This project is licensed under the LICENSE License.

