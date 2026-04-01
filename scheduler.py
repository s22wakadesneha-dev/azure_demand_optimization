"""
Milestone 4 - Task 3: Automated Scheduler
Runs the batch prediction pipeline daily at 06:00 UTC.
Also implements threshold-based forecast alerting.
"""

import schedule
import time
import logging
import os
from datetime import datetime

# ----------------------------------------
# Logging Setup
# ----------------------------------------
logging.basicConfig(
    filename="scheduler_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------
# Config
# ----------------------------------------
ALERT_THRESHOLD_PCT = 0.80      # Alert if forecast > 80% of provisioned capacity
SCHEDULE_TIME_UTC = "06:00"     # Daily run time in UTC


# ----------------------------------------
# Import batch prediction
# ----------------------------------------
def run_batch_pipeline():
    """
    Executes the full batch prediction pipeline.
    Logs success or failure with timestamps.python batch_predict.py
    """
    run_time = datetime.utcnow().isoformat()
    print(f"\n[{run_time} UTC] ▶ Starting batch prediction pipeline...")
    logging.info(f"Pipeline started at {run_time}")

    try:
        # Import and run batch predict
        import batch_predict
        batch_predict.main()

        logging.info(f"Pipeline completed successfully at {datetime.utcnow().isoformat()}")
        print(f"✅ Pipeline completed at {datetime.utcnow().isoformat()} UTC")

        # Run threshold alerting
        check_forecast_thresholds()

    except Exception as e:
        logging.error(f"Pipeline FAILED: {e}")
        print(f"❌ Pipeline FAILED: {e}")


# ----------------------------------------
# Threshold Alerting
# ----------------------------------------
def check_forecast_thresholds():
    """
    Reads forecast_output.csv and alerts if predicted usage
    exceeds ALERT_THRESHOLD_PCT of provisioned capacity.
    """
    import pandas as pd

    FORECAST_FILE = "forecast_output.csv"

    if not os.path.exists(FORECAST_FILE):
        logging.warning("forecast_output.csv not found — skipping threshold check.")
        return

    df = pd.read_csv(FORECAST_FILE)

    if 'predicted_usage' not in df.columns or 'provisioned_capacity' not in df.columns:
        logging.warning("Missing columns for threshold check.")
        return

    df['utilisation_ratio'] = df['predicted_usage'] / df['provisioned_capacity'].replace(0, float('nan'))
    breaches = df[df['utilisation_ratio'] > ALERT_THRESHOLD_PCT]

    if not breaches.empty:
        alert_msg = (
            f"⚠️  CAPACITY ALERT: {len(breaches)} forecast records exceed "
            f"{ALERT_THRESHOLD_PCT*100:.0f}% of provisioned capacity.\n"
            f"   Regions affected: {breaches['region'].unique().tolist()}\n"
            f"   Max utilisation: {breaches['utilisation_ratio'].max():.1%}\n"
            f"   Action: Review provisioning for upcoming period."
        )
        print(alert_msg)
        logging.warning(alert_msg)
    else:
        print("✅ No capacity threshold breaches detected.")
        logging.info("Threshold check passed — no breaches.")


# ----------------------------------------
# Main Scheduler Loop
# ----------------------------------------
def main():
    print("=" * 55)
    print("  Azure Demand Forecast — Automated Scheduler")
    print(f"  Schedule: Daily at {SCHEDULE_TIME_UTC} UTC")
    print(f"  Alert Threshold: {ALERT_THRESHOLD_PCT*100:.0f}% of provisioned capacity")
    print("=" * 55)
    print("Scheduler is running. Press Ctrl+C to stop.\n")

    logging.info(f"Scheduler started. Runs daily at {SCHEDULE_TIME_UTC} UTC.")

    # Schedule the daily pipeline
    schedule.every().day.at(SCHEDULE_TIME_UTC).do(run_batch_pipeline)

    # Optional: run immediately on startup for testing
    print("Running pipeline immediately (startup run)...")
    run_batch_pipeline()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(30)     # Check every 30 seconds


if __name__ == "__main__":
    main()
