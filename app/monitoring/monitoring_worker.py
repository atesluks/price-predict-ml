import sys
import os
from datetime import datetime, UTC

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
from app.database import SessionLocal
from app.utils.cache_price import get_all_cached_price_records
from app.utils.data import get_latest_hourly_records_for_pairs, get_latest_daily_records_for_pairs
from app.utils.misc import is_valid_number, alert_slack, log, log_error, print_log
from app.utils.load_file import get_latest_run_from_mlflow

# Constants
ENVIRONMENT=os.getenv('ENVIRONMENT')
SLACK_WEBHOOK_URL=os.getenv('SLACK_WEBHOOK_URL')

MODEL_TRAINING_HOURLY_DATA_ALERT_THRESHOLD_HOURS = int(os.getenv('MODEL_TRAINING_HOURLY_DATA_ALERT_THRESHOLD_HOURS', 2))  # Maximum allowed age (in hours) for hourly training data before triggering a Slack alert
MODEL_TRAINING_DAILY_DATA_ALERT_THRESHOLD_HOURS = int(os.getenv('MODEL_TRAINING_DAILY_DATA_ALERT_THRESHOLD_HOURS', 26))  # Maximum allowed age (in hours) for daily training data before triggering a Slack alert

MONITORING_SCHEDULER_INTERVAL_MINUTES = 10 # How often the monitoring scheduler runs (in minutes). The scheduler will check cache prices, data freshness, and model training status at this interval
CACHE_PRICE_ALERT_THRESHOLD_MINUTES = 10 # If the cache price has not been updated for this many minutes, raise slack alert
HOURLY_DATA_ALERT_THRESHOLD_HOURS = 2  # Maximum allowed age (in hours) for hourly market data before triggering a Slack alert
DAILY_DATA_ALERT_THRESHOLD_HOURS = 26  # Maximum allowed age (in hours) for daily market data before triggering a Slack alert

MODELS = [
    {"model": "06_btc_1m_from_btc_8h_multi", "experiment": "1 month BTC prediction", "data_type": "hourly"},
    {"model": "07_btc_2m_from_btc_8h_multi", "experiment": "2 months BTC prediction", "data_type": "hourly"},
    {"model": "08_btc_3m_from_btc_7d_multi", "experiment": "3 months BTC prediction", "data_type": "daily"},
    {"model": "09_btc_6m_from_btc_7d_multi", "experiment": "6 months BTC prediction", "data_type": "daily"},
    {"model": "10_eth_1m_from_eth_8h_multi", "experiment": "1 month ETH prediction", "data_type": "hourly"},
    {"model": "11_eth_2m_from_eth_8h_multi", "experiment": "2 months ETH prediction", "data_type": "hourly"},
    {"model": "12_eth_3m_from_eth_7d_multi", "experiment": "3 months ETH prediction", "data_type": "daily"},
    {"model": "13_eth_6m_from_eth_7d_multi", "experiment": "6 months ETH prediction", "data_type": "daily"},
]

DATA_ASSET_PAIRS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
]

WORKER_NAME = "monitoring"

# Checking if the chache worker works well (if all the prices are cached on time)
def check_cache():
    service_name = "cache"
    session = SessionLocal()
    
    try:
        # Get cached prices from DB
        price_records = get_all_cached_price_records(session)

        # Iterate through the array of models
        for model in MODELS:
            model_name = model['model']
            
            # Checking if there is cache price for this model in the DB at all
            if model_name not in price_records or not price_records[model_name]:
                alert_slack(WORKER_NAME, service_name, f"No record in cache_price table for model {model_name}")
                continue
            
            # Check if the cache price is not a number (NaN or None)
            price = price_records[model_name]["price"]
            if not is_valid_number(price):
                alert_slack(WORKER_NAME, service_name, f"Invalid price for model {model_name}")
                continue
            
            # Check if the difference between now and the updated_at is below configured threshold
            updated_at = price_records[model_name]["updated_at"].replace(tzinfo=UTC)
            time_now = datetime.now(UTC)
            time_diff = time_now - updated_at
            time_diff_minutes = time_diff.total_seconds() / 60
            
            if time_diff_minutes > CACHE_PRICE_ALERT_THRESHOLD_MINUTES:
                alert_slack(WORKER_NAME, service_name, f"Price for model {model_name} has not been updated in the last {CACHE_PRICE_ALERT_THRESHOLD_MINUTES} minutes")
    finally:
        session.close()
    
    log(WORKER_NAME, f"[{service_name}]  Done checking")

# Checking if the data worker works well (if hourly and daily data was fetched on time)
def check_data_workers():
    service_name = "data_workers"
    session = SessionLocal()
    
    try:
        # Checking records for hourly data
        latest_records = get_latest_hourly_records_for_pairs(session, DATA_ASSET_PAIRS)
        check_data_workers_impl(service_name, latest_records, "DataCGCoinsMarketChart1h", HOURLY_DATA_ALERT_THRESHOLD_HOURS)
        
        # Checking records for daily data
        latest_records = get_latest_daily_records_for_pairs(session, DATA_ASSET_PAIRS)
        check_data_workers_impl(service_name, latest_records, "DataCGCoinsMarketChart1d", DAILY_DATA_ALERT_THRESHOLD_HOURS)
    finally:
        session.close()
    
    log(WORKER_NAME, f"[{service_name}]  Done checking")

def check_data_workers_impl(service_name, latest_records, data_table, time_interval_hours) -> None:
    for pair in DATA_ASSET_PAIRS:
        if pair not in latest_records or not latest_records[pair]:
            alert_slack(WORKER_NAME, service_name, f"No data for {pair} in {data_table}")
            continue
        
        price = latest_records[pair]["price"]
        mcap = latest_records[pair]["mcap"]
        volume = latest_records[pair]["volume"]
        time = latest_records[pair]["time"].replace(tzinfo=UTC)

        # Check if the price is a valid number
        if not is_valid_number(price):
            alert_slack(WORKER_NAME, service_name, f"Invalid price for {pair} in {data_table}")
            continue

        # Check if the mcap is a valid number
        if not is_valid_number(mcap):
            alert_slack(WORKER_NAME, service_name, f"Invalid mcap for {pair} in {data_table}")
            continue

        # Check if the volume is a valid number
        if not is_valid_number(volume):
            alert_slack(WORKER_NAME, service_name, f"Invalid volume for {pair} in {data_table}")
            continue
        
        # Check if the difference between now and the updated_at is below configured threshold
        time_now = datetime.now(UTC)
        time_diff = time_now - time
        time_diff_hours = time_diff.total_seconds() / 3600

        if time_diff_hours > time_interval_hours:
            alert_slack(WORKER_NAME, service_name, f"Market data for {pair} was not updated for {time_interval_hours} hours in {data_table}")

# Checking if the ML model training jobs works well (that each ML model got trained shortly after new data became available)
def check_model_training():
    service_name = "model_training"

    for model in MODELS:
        try:
            model_name = model["model"]

            # Retrieving and checking latest run
            run_info = get_latest_run_from_mlflow(model["experiment"], model_name)
            if not run_info:
                alert_slack(WORKER_NAME, service_name, f"Experiment or run for {model_name} not found")
                continue

            # Check if the MSE is a valid number
            mse = run_info["mse"]
            if not is_valid_number(mse):
                alert_slack(WORKER_NAME, service_name, f"Invalid MSE for {model_name}")
                continue
            
            # Check if the finished_time is valid and if the difference between it and now is below configured threshold
            time = run_info["finished_time"]
            if not is_valid_number(time):
                alert_slack(WORKER_NAME, service_name, f"Invalid time for {model_name}")

            time_now = datetime.now(UTC)
            time_as_datetime = datetime.fromtimestamp(time / 1000, tz=UTC)
            time_diff = time_now - time_as_datetime
            time_diff_hours = time_diff.total_seconds() / 3600

            time_interval_hours = MODEL_TRAINING_HOURLY_DATA_ALERT_THRESHOLD_HOURS if model["data_type"] == "hourly" else MODEL_TRAINING_DAILY_DATA_ALERT_THRESHOLD_HOURS
            
            if time_diff_hours > time_interval_hours:
                alert_slack(WORKER_NAME, service_name, f"Model {model_name} was not updated for {time_interval_hours} hours")

        except Exception as e:
            log_error(WORKER_NAME, f"Error in check_model_training(): {e}")

    log(WORKER_NAME, f"[{service_name}]  Done checking")

# Main method that launches the scheduler
def scheduled_task() -> None:
    check_cache()
    check_data_workers()
    check_model_training()

# Main method that launches the scheduler
if __name__ == "__main__":
    # Run the task immediately instead of waiting for the first scheduler cycle (10 minutes)
    log(WORKER_NAME, message="Starting scheduler")
    scheduled_task()

    # Create an instance of the APScheduler scheduler and schedule the task to run every minute
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_task, IntervalTrigger(minutes=MONITORING_SCHEDULER_INTERVAL_MINUTES))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass