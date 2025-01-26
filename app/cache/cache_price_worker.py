import sys
import os
from datetime import datetime, UTC

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
from app.database import SessionLocal
from app.utils.cache_price import get_price
from app.utils.misc import log, log_error, print_log

# How often the cache scheduler runs (in minutes). The scheduler will run ML models for price prediction and save in cache DB table.
CACHE_SCHEDULER_INTERVAL_MINUTES = 1 
WORKER_NAME = "cache_worker"

MODELS = [
    {"model": "06_btc_1m_from_btc_8h_multi", "pair": "BTC-USD", "lags": 8, "data_type": "hourly"},
    {"model": "07_btc_2m_from_btc_8h_multi", "pair": "BTC-USD", "lags": 8, "data_type": "hourly"},
    {"model": "08_btc_3m_from_btc_7d_multi", "pair": "BTC-USD", "lags": 7, "data_type": "daily"},
    {"model": "09_btc_6m_from_btc_7d_multi", "pair": "BTC-USD", "lags": 7, "data_type": "daily"},
    {"model": "10_eth_1m_from_eth_8h_multi", "pair": "ETH-USD", "lags": 8, "data_type": "hourly"},
    {"model": "11_eth_2m_from_eth_8h_multi", "pair": "ETH-USD", "lags": 8, "data_type": "hourly"},
    {"model": "12_eth_3m_from_eth_7d_multi", "pair": "ETH-USD", "lags": 7, "data_type": "daily"},
    {"model": "13_eth_6m_from_eth_7d_multi", "pair": "ETH-USD", "lags": 7, "data_type": "daily"},
]

def scheduled_task():
    print_log(WORKER_NAME, "Starting scheduled task")
    session = SessionLocal()
    
    try:
        for model in MODELS:
            model_name = model['model']
            pair = model['pair']
            lags = model['lags']
            data_type = model['data_type']

            get_price(session, model_name, pair, lags, data_type)
        
        log(WORKER_NAME, f"Finished caching all")
    
    except Exception as e:
        log_error(WORKER_NAME, f"Error in scheduled_task(): {e}")
    finally:
        session.close()

# Main method that launches the scheduler
if __name__ == "__main__":
    # Run the task immediately instead of waiting for the first scheduler cycle (1 minute)
    log(WORKER_NAME, message="Starting scheduler")
    scheduled_task()

    # Create an instance of the APScheduler scheduler and schedule the task to run every minute
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_task, IntervalTrigger(minutes=CACHE_SCHEDULER_INTERVAL_MINUTES))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass