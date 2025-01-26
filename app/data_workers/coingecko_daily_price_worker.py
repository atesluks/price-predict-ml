# Retrieves daily market data for an asset (price, market cap, 24-h volume). 
# Manually processing last-day data so it fits 1-hour time series, since it is 5-min for the last day. 
# More: https://docs.coingecko.com/reference/coins-id-market-chart

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, UTC
import requests
import math
from sqlalchemy.orm import Session
from app.models import DataCGCoinsMarketChart1d
from app.utils.model_training_queue import add_daily_data_dependent_models
from app.utils.misc import log, log_error, print_log, get_env_var
from app.utils.data import COINGECKO_IDS
from dotenv import load_dotenv
from app.database import SessionLocal

load_dotenv()

WORKER_NAME = "coingecko_daily_price_worker"

# Returns date and time of the latest data record
def get_time_last_daily_record(session: Session, asset_pair):
    last_record = session.query(DataCGCoinsMarketChart1d).filter(DataCGCoinsMarketChart1d.pair == asset_pair).order_by(DataCGCoinsMarketChart1d.time.desc()).first()
    if last_record and last_record.time:
        return last_record.time.replace(tzinfo=UTC)
    return None

def fetch_daily_data_from_coingecko(days, asset_pair):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{COINGECKO_IDS[asset_pair]}/market_chart?vs_currency=usd&days={days}&interval=daily"
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": get_env_var("COINGECKO_API")
    }
    response = requests.get(url, headers=headers)
    return response.json()

def process_and_store_data(session: Session, asset_pair, data):
    # Remove last component (because it is the latest data)
    # Example: Latest prices in the array are: [..., Jan 23 2025 00:00:00, Jan 24 2025 00:00:00, Jan 24 2025 11:43:58]
    # We need the daily data without the current data, therefore removing last element
    data["prices"].pop()
    data["market_caps"].pop()
    data["total_volumes"].pop()

    added_records = 0

    # Record all other data points
    for point in data["prices"]:
        price = point[1]
        mcap = next((item[1] for item in data["market_caps"] if item[0] == point[0]), None)
        volume = next((item[1] for item in data["total_volumes"] if item[0] == point[0]), None)
        timestamp = datetime.fromtimestamp(point[0] / 1000, UTC)

        # Double checking that there is no data point with such time already
        if not session.query(DataCGCoinsMarketChart1d).filter_by(time=timestamp, pair=asset_pair).first():
            new_record = DataCGCoinsMarketChart1d(
                pair=asset_pair,
                time=timestamp,
                price=price,
                mcap=mcap,
                volume=volume,
                updated_at=datetime.now(UTC)
            )
            session.add(new_record)
            session.commit()
            added_records = added_records + 1

    return added_records

def retrieve_data(asset_pair):
    print_log(WORKER_NAME, f"Starting data collection for {asset_pair}")
    
    session = SessionLocal()
    added_records = 0

    try:
        last_time = get_time_last_daily_record(session, asset_pair)
        if last_time:
            # Calculating for how many days we need to query the data (based on latest recorded data)
            days_since_last_record = math.floor((datetime.now(UTC) - last_time).total_seconds() / 86400)
        else:
            # By default querying data for the last 3 years
            days_since_last_record = 1095 # 3 years

        if(days_since_last_record > 0):
            data = fetch_daily_data_from_coingecko(days_since_last_record, asset_pair)
            added_records = process_and_store_data(session, asset_pair, data)
            log(WORKER_NAME, f"Saved {added_records} new records for {asset_pair} for the past {days_since_last_record} days")
        else:
            log(WORKER_NAME, f"No new records for {asset_pair}")
            # log(WORKER_NAME, f"Nothing to add for {asset_pair}. Last fetched on {last_time}, now {datetime.now(UTC)}") # For debugging
        
    except Exception as e:
        log_error(WORKER_NAME, f"Error in retrieve_data() for pair {asset_pair}: {str(e)}")
    finally:
        session.close()

    return added_records

# Function to be scheduled
def scheduled_task():
    print_log(WORKER_NAME, "Starting scheduled task")

    try:
        # Retrieving the data
        total_datapoints_retrieved = 0

        for pair in COINGECKO_IDS.keys():
            result = retrieve_data(pair)
            total_datapoints_retrieved = total_datapoints_retrieved + result

        # Add dependent ML models to a training queue if there is new data
        if total_datapoints_retrieved > 0:
            add_daily_data_dependent_models(WORKER_NAME)
            log(WORKER_NAME, "Added dependent ML models to training queue")

        print_log(WORKER_NAME, "Finished scheduled task")

    except Exception as e:
        log_error(WORKER_NAME, f"Error in scheduled_task(): {e}")

# Main method that launches the scheduler
if __name__ == "__main__":
    # Run the task immediately instead of waiting for the first scheduler cycle (30 minutes)
    log(WORKER_NAME, message="Starting scheduler")
    scheduled_task()

    # Create an instance of the APScheduler scheduler and schedule the task to run every 30 minutes
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_task, IntervalTrigger(minutes=30))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass