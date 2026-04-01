"""
Milestone 4 - Task 4: Monitoring & Retraining Pipeline
Tracks model RMSE over time and triggers retraining when drift is detected.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Logging Setup
# ----------------------------------------
logging.basicConfig(
    filename="monitoring_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------
# Config
# ----------------------------------------
MODEL_PATH         = "best_arima_model.pkl"
RETRAINED_PATH     = "best_arima_model_retrained.pkl"
REGISTRY_PATH      = "model_registry.json"
UPDATED_DATA_CSV   = "updated_data.csv"     # New data for retraining
FORECAST_CSV       = "forecast_output.csv"  # Produced by batch_predict.py
RMSE_LOG_CSV       = "rmse_history.csv"
RMSE_ALERT_MARGIN  = 0.20                   # Alert if RMSE grows >20% above baseline
BASELINE_RMSE      = 130.0                  # Set from Milestone 3 best result


# ----------------------------------------
# 1. Compute RMSE from Forecast Output
# ----------------------------------------
def compute_monitoring_rmse(forecast_csv: str) -> dict:
    """
    Compares predicted_usage vs usage_units in forecast_output.csv.
    Returns a dict of metrics.
    """
    if not os.path.exists(forecast_csv):
        raise FileNotFoundError(f"'{forecast_csv}' not found. Run batch_predict.py first.")

    df = pd.read_csv(forecast_csv)

    if 'usage_units' not in df.columns or 'predicted_usage' not in df.columns:
        raise ValueError("forecast_output.csv must contain 'usage_units' and 'predicted_usage' columns.")

    actual    = df['usage_units'].dropna()
    predicted = df['predicted_usage'].dropna()

    # Align lengths
    n = min(len(actual), len(predicted))
    actual, predicted = actual.iloc[:n], predicted.iloc[:n]

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    bias = float(np.mean(predicted.values - actual.values))

    # Directional accuracy
    actual_diff    = np.diff(actual.values)
    predicted_diff = np.diff(predicted.values)
    dir_acc = np.mean(np.sign(actual_diff) == np.sign(predicted_diff)) * 100

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "directional_accuracy_pct": round(dir_acc, 2),
    }

    return metrics


# ----------------------------------------
# 2. Log RMSE History
# ----------------------------------------
def log_rmse(metrics: dict):
    """Appends RMSE metrics to a CSV log for trend tracking."""
    row = pd.DataFrame([metrics])
    if os.path.exists(RMSE_LOG_CSV):
        row.to_csv(RMSE_LOG_CSV, mode='a', header=False, index=False)
    else:
        row.to_csv(RMSE_LOG_CSV, index=False)
    logging.info(f"RMSE logged: {metrics}")
    print(f"📋 RMSE logged to {RMSE_LOG_CSV}")


# ----------------------------------------
# 3. Alert Check
# ----------------------------------------
def check_alert(rmse: float, baseline: float, margin: float) -> bool:
    """Returns True if RMSE has exceeded the alert threshold."""
    threshold = baseline * (1 + margin)
    if rmse > threshold:
        msg = (
            f"⚠️  MODEL DRIFT ALERT!\n"
            f"   Current RMSE : {rmse:.2f}\n"
            f"   Baseline RMSE: {baseline:.2f}\n"
            f"   Threshold    : {threshold:.2f} ({margin*100:.0f}% above baseline)\n"
            f"   → Retraining is recommended."
        )
        print(msg)
        logging.warning(msg)
        return True
    else:
        print(f"✅ RMSE within limits: {rmse:.2f} (threshold: {threshold:.2f})")
        logging.info(f"RMSE OK: {rmse:.2f} (threshold: {threshold:.2f})")
        return False


# ----------------------------------------
# 4. Retraining Pipeline
# ----------------------------------------
def retrain_model(data_path: str) -> dict:
    """
    Retrains the ARIMA model on updated data.
    Only replaces the production model if accuracy improves.
    """
    print("\n🔄 Starting retraining pipeline...")
    logging.info("Retraining pipeline started.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Updated data not found: '{data_path}'")

    # Load & preprocess updated data
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['usage_units'] = pd.to_numeric(df['usage_units'], errors='coerce').interpolate()

    y = df['usage_units'].dropna()
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]

    if len(y_train) < 20:
        raise ValueError("Not enough data to retrain (need at least 20 training rows).")

    # Re-fit ARIMA (use same best order as original training or search)
    print("Fitting new ARIMA model...")
    best_rmse_new = float('inf')
    best_order = (1, 1, 1)

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    m = ARIMA(y_train, order=(p, d, q)).fit()
                    pred = m.forecast(steps=len(y_test))
                    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                    if rmse < best_rmse_new:
                        best_rmse_new = rmse
                        best_order = (p, d, q)
                except Exception:
                    continue

    # Fit final model on full training data
    new_model = ARIMA(y_train, order=best_order).fit()
    new_pred  = new_model.forecast(steps=len(y_test))
    new_rmse  = float(np.sqrt(mean_squared_error(y_test, new_pred)))

    print(f"   Best order: {best_order} | New RMSE: {new_rmse:.2f}")

    # Compare with current production model
    old_model  = joblib.load(MODEL_PATH)
    old_pred   = old_model.forecast(steps=len(y_test))
    old_rmse   = float(np.sqrt(mean_squared_error(y_test, old_pred)))

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "arima_order": str(best_order),
        "rmse_before": round(old_rmse, 4),
        "rmse_after":  round(new_rmse, 4),
        "n_rows_used": len(y),
        "deployed": False,
    }

    # Only deploy if better
    if new_rmse < old_rmse:
        joblib.dump(new_model, RETRAINED_PATH)
        # Replace production model
        joblib.dump(new_model, MODEL_PATH)
        result["deployed"] = True
        msg = (
            f"✅ New model deployed!\n"
            f"   RMSE improved: {old_rmse:.2f} → {new_rmse:.2f}"
        )
        print(msg)
        logging.info(msg)
    else:
        msg = (
            f"⚠️  New model NOT deployed — no improvement.\n"
            f"   Old RMSE: {old_rmse:.2f} | New RMSE: {new_rmse:.2f}"
        )
        print(msg)
        logging.warning(msg)

    # Update model registry
    update_registry(result)
    return result


# ----------------------------------------
# 5. Model Registry
# ----------------------------------------
def update_registry(record: dict):
    """Maintains a JSON log of all retraining events."""
    registry = []
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
    registry.append(record)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"📁 Registry updated: {REGISTRY_PATH}")
    logging.info(f"Registry updated with retraining record.")


# ----------------------------------------
# 6. Main Monitoring Run
# ----------------------------------------
def run_monitoring(trigger_retrain=False):
    print("=" * 55)
    print("  Azure Demand Forecast — Monitoring Pipeline")
    print(f"  Time: {datetime.utcnow().isoformat()} UTC")
    print("=" * 55)

    try:
        # Step 1: Compute current RMSE
        metrics = compute_monitoring_rmse(FORECAST_CSV)
        print(f"\n📊 Current Metrics:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        # Step 2: Log metrics
        log_rmse(metrics)

        # Step 3: Alert check
        drift_detected = check_alert(
            rmse=metrics['rmse'],
            baseline=BASELINE_RMSE,
            margin=RMSE_ALERT_MARGIN
        )

        # Step 4: Retrain if drift detected or forced
        if drift_detected or trigger_retrain:
            if os.path.exists(UPDATED_DATA_CSV):
                retrain_result = retrain_model(UPDATED_DATA_CSV)
                print(f"\n📄 Retraining Result: {retrain_result}")
            else:
                print(f"⚠️  Retraining skipped — '{UPDATED_DATA_CSV}' not found.")
                logging.warning(f"Retraining skipped: {UPDATED_DATA_CSV} not found.")

        logging.info("Monitoring run completed.")
        print("\n✅ Monitoring run complete.")

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        logging.error(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Monitoring error: {e}")


# ----------------------------------------
# Entry Point
# ----------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Azure Demand Forecast — Monitoring & Retraining")
    parser.add_argument('--retrain', action='store_true', help='Force retraining regardless of RMSE')
    args = parser.parse_args()

    run_monitoring(trigger_retrain=args.retrain)
