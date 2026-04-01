"""
Milestone 4 - Task 1: FastAPI Prediction API
Azure Demand Forecasting - Real-Time Prediction Service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

# ----------------------------------------
# Setup Logging
# ----------------------------------------
logging.basicConfig(
    filename="api_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Azure Demand Forecast API",
    description="Real-time prediction API for Azure resource demand forecasting",
    version="1.0.0"
)

# ----------------------------------------
# Load Model at Startup
# ----------------------------------------
MODEL_PATH = "best_arima_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please train and save the model first.")
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
    print("✅ Model loaded successfully.")

# ----------------------------------------
# Request Schema
# ----------------------------------------
class PredictRequest(BaseModel):
    timestamp: str                       # e.g. "2024-01-15 10:00:00"
    region: str                          # e.g. "East-US"
    service_type: str                    # e.g. "Compute"
    provisioned_capacity: float
    cost_usd: float
    availability_pct: float
    economic_growth_index: float
    marketing_index: float
    it_spending_growth: float
    is_holiday: int                      # 0 or 1
    steps: int = 1                       # number of forecast steps ahead

class PredictResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    forecast: list
    model_used: str
    timestamp_requested: str
    steps: int

# ----------------------------------------
# Helper: Preprocess Input
# ----------------------------------------
def preprocess_input(data: PredictRequest) -> pd.DataFrame:
    """
    Converts request data into a preprocessed DataFrame.
    Column names are normalised to match training format.
    """
    # Normalise region and service_type (same as training)
    region = data.region.strip().replace(" ", "-")
    region_map = {
        'central-india': 'Central-India',
        'west-us': 'West-US',
        'east-us': 'East-US',
        'east-asia': 'East-Asia',
        'uk-south': 'UK-South',
    }
    region = region_map.get(region.lower(), region)

    ts = pd.to_datetime(data.timestamp, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp format: '{data.timestamp}'. Use 'YYYY-MM-DD HH:MM:SS'.")

    row = {
        'year': ts.year,
        'month': ts.month,
        'day': ts.day,
        'hour': ts.hour,
        'day_of_week': ts.dayofweek,
        'provisioned_capacity': data.provisioned_capacity,
        'cost_usd': data.cost_usd,
        'availability_pct': data.availability_pct,
        'economic_growth_index': data.economic_growth_index,
        'marketing_index': data.marketing_index,
        'it_spending_growth': data.it_spending_growth,
        'is_holiday': data.is_holiday,
        # One-hot encoding for region (drop_first = Central-India)
        'region_ East Asia': 1 if region == 'East-Asia' else 0,
        'region_ East US': 1 if region == 'East-US' else 0,
        'region_ UK South': 1 if region == 'UK-South' else 0,
        'region_ West US': 1 if region == 'West-US' else 0,
        # One-hot encoding for service_type (drop_first = Compute)
        'service_type_ Storage': 1 if data.service_type.strip().lower() == 'storage' else 0,
    }
    return pd.DataFrame([row])

# ----------------------------------------
# POST /predict — Real-Time Prediction
# ----------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        # For ARIMA: forecast N steps ahead
        forecast_values = model.forecast(steps=request.steps)
        forecast_list = [round(float(v), 4) for v in forecast_values]

        logging.info(f"Prediction made | region={request.region} | steps={request.steps} | forecast={forecast_list}")

        return PredictResponse(
            forecast=forecast_list,
            model_used="ARIMA (best saved model)",
            timestamp_requested=request.timestamp,
            steps=request.steps
        )

    except ValueError as ve:
        logging.error(f"Input validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ----------------------------------------
# GET /health — Health Check
# ----------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# ----------------------------------------
# GET / — Welcome
# ----------------------------------------
@app.get("/")
def root():
    return {
        "message": "Azure Demand Forecast API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }

# ----------------------------------------
# Run: uvicorn api:app --reload
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
