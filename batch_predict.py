"""
Milestone 4 - Task 1: Batch Prediction Script
Reads new_data.csv → applies preprocessing → writes forecast_output.csv
"""

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
    filename="batch_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------
# Config
# ----------------------------------------
INPUT_CSV = "new_data.csv"          # Input: new usage data
OUTPUT_CSV = "forecast_output.csv"  # Output: predictions written here
MODEL_PATH = "best_arima_model.pkl" # Saved model from Milestone 3

# ----------------------------------------
# Load Model
# ----------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Train and save the model first.")
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded for batch prediction.")
    return model

# ----------------------------------------
# Load & Validate Input Data
# ----------------------------------------
def load_input_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Check required columns
    required = ['timestamp', 'region', 'service_type', 'usage_units',
                 'provisioned_capacity', 'cost_usd', 'availability_pct',
                 'economic_growth_index', 'marketing_index', 'it_spending_growth', 'is_holiday']

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Handle missing values
    if df['usage_units'].isnull().any():
        logging.warning("Missing values in 'usage_units' — interpolating.")
        df['usage_units'] = df['usage_units'].interpolate()

    df['cost_usd'] = df['cost_usd'].fillna(df['usage_units'] * 0.5)
    df['availability_pct'] = df['availability_pct'].ffill()
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

    # Normalise region
    df['region'] = df['region'].str.strip().str.lower().str.replace(" ", "-")
    region_map = {
        'central-india': 'Central-India',
        'west-us': 'West-US',
        'east-us': 'East-US',
        'east-asia': 'East-Asia',
        'uk-south': 'UK-South',
    }
    df['region'] = df['region'].replace(region_map)

    logging.info(f"Loaded {len(df)} rows from '{path}'")
    return df

# ----------------------------------------
# Run Batch Prediction (ARIMA)
# ----------------------------------------
def run_batch_prediction(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    For ARIMA: forecast N steps = number of rows in df.
    Append prediction column and relevant metadata.
    """
    n = len(df)
    logging.info(f"Forecasting {n} steps ahead using ARIMA...")

    forecast_values = model.forecast(steps=n)

    df['predicted_usage'] = forecast_values.values
    df['forecast_generated_at'] = datetime.utcnow().isoformat()

    # Keep only useful output columns
    output_cols = [
        'timestamp', 'region', 'service_type',
        'usage_units', 'predicted_usage',
        'provisioned_capacity', 'cost_usd', 'availability_pct',
        'forecast_generated_at'
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols]

# ----------------------------------------
# Save Output
# ----------------------------------------
def save_output(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logging.info(f"Forecast saved to '{path}' ({len(df)} rows)")
    print(f"✅ Forecast saved: {path} ({len(df)} rows)")

# ----------------------------------------
# Main
# ----------------------------------------
def main():
    print("=" * 50)
    print("  Azure Demand Batch Prediction")
    print(f"  Run at: {datetime.utcnow().isoformat()} UTC")
    print("=" * 50)

    try:
        model = load_model()
        df = load_input_data(INPUT_CSV)
        result = run_batch_prediction(model, df)
        save_output(result, OUTPUT_CSV)

        print("\nSample Output:")
        print(result.head())

        logging.info("Batch prediction completed successfully.")

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        logging.error(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"❌ Data Error: {e}")
        logging.error(f"ValueError: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
