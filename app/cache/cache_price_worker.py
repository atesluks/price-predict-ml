import sys
import os
from datetime import datetime, UTC

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.abspath(os.path.join("..", "..", "..")))
from app.database import SessionLocal
from app.utils.cache_price import get_price
from app.utils.misc import log, log_error, print_log, get_model_configs, parse_timeframe

# How often the cache scheduler runs (in minutes). The scheduler will run ML models for price prediction and save in cache DB table.
CACHE_SCHEDULER_INTERVAL_MINUTES = 1 
WORKER_NAME = "cache_worker"

# Load models configuration once at module level
MODELS = get_model_configs()

def scheduled_task():
    print_log(WORKER_NAME, "Starting scheduled task")
    session = SessionLocal()
    
    try:
        for model in MODELS:
            model_name = model["name"]
            pair = model["pair"]
            lags, data_type = parse_timeframe(model["input_timeframe"])

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