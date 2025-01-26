"""
Description: Predicting BTC price for the next 6 months from the last 7 days (daily) market data.
Model: Multi-model.
Input: Last 7 days of daily data of BTC price and 24-volume.
Output: Predicted BTC price in the next 6 months.
"""

### 0. Importing libraries and modules
import sys
import os
import time

import mlflow
import logging

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
from app.utils.model_training_queue import should_start_training, set_model_status_training, set_model_status_done
from app.ml_models.utils import load_daily_data_from_db, model_eval, retraining_whole_model, logging_results, data_pre_processing, training_model

# Constants
ASSET_PAIR = "BTC-USD"
MODEL_NAME = "09_btc_6m_from_btc_7d_multi"
MODEL_DESCRIPTION = "Predicting BTC price for the next 6 months from the last 7 day (daily) market data."
INPUT_DESCRIPTION = "Last 7 days of daily data of BTC price and 24-volume."
OUTPUT_DESCRIPTION = "Predicted BTC price in the next 6 months."

SOURCE_NAME = "data_cg_coins_market_chart_1d"
SOURCE_COLUMNS = ["Time", "Price", "24h_Volume"]

RETRAIN_INTERVAL = "3 hours"
N_TEST = 92 # Number of testing data points
LAGS = 7

OPTUNA_TRIALS = 30
RANDOM_SEARCH_N_ITER = 10
PRUNER_PATIENCE = 5

DAYS_AHEAD_TO_PREDICT = 180

mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_URL'))
mlflow.set_experiment("6 months BTC prediction")

def scheduled_task():
    # Check if model should start training. If not, exiting.
    if not should_start_training(MODEL_NAME):
        return
    
    set_model_status_training(MODEL_NAME)
    
    # 0. Starting MLFlow run
    start_time_general = time.time()
    mlflow.start_run()

    # 1. loading the data
    df = load_daily_data_from_db(MODEL_NAME, ASSET_PAIR)

    # 2. Pre-processing the data and splitting to training and test sets
    n_train, X, y, X_train_scaled, y_train, X_test_scaled, y_test = data_pre_processing(df, MODEL_NAME, LAGS, DAYS_AHEAD_TO_PREDICT, N_TEST)

    # 3. Testing different models and taking the best one (heaviest part)
    (best_model_name, 
    model, 
    fresh_model, 
    random_search_all_model_variations_by_score, 
    random_search_all_models_by_time,
    random_search_top_10_models_by_score, 
    optimization_results_df,
    random_search_total_time,
    random_search_top10_total_time,
    bayes_opt_total_time) = training_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # 4. Evaluating the best model
    mse, mae, rmse, mape, r2, y_pred = model_eval(MODEL_NAME, model, X_test_scaled, y_test)

    # 5. Retraining the best model on the whole dataset
    scaler, model = retraining_whole_model(fresh_model, X, y)

    # 6. Logging and saving the model to MLFlow
    params = {
        "model_name": MODEL_NAME,
        "model_description": MODEL_DESCRIPTION,
        "start_time_general": start_time_general,
        "best_model_name": best_model_name,
        "lags": LAGS,
        "input_description": INPUT_DESCRIPTION,
        "output_description": OUTPUT_DESCRIPTION,
        "n_train": n_train,
        "n_test": N_TEST,
        "source_name": SOURCE_NAME,
        "source_columns": SOURCE_COLUMNS,
        "data_from": df.index.min().isoformat(),
        "data_to": df.index.max().isoformat(),
        "random_search_total_time": random_search_total_time,
        "random_search_top10_total_time": random_search_top10_total_time,
        "bayes_opt_total_time": bayes_opt_total_time,
    }

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }

    logging_results(
        params, 
        metrics, 
        scaler, 
        model,
        X_train_scaled, 
        y_pred,
        random_search_all_model_variations_by_score, 
        random_search_all_models_by_time,
        random_search_top_10_models_by_score, 
        optimization_results_df
    )

    # 7. Ending the MLFlow run
    mlflow.end_run()

    set_model_status_done(MODEL_NAME)

if __name__ == "__main__":
    # Create an instance of the APPScheduler scheduler
    logging.getLogger('apscheduler').setLevel(logging.ERROR)
    scheduler = BlockingScheduler()

    # Schedule the task to run every hour with model_db_id as an argument
    scheduler.add_job(scheduled_task, IntervalTrigger(seconds=10), max_instances=1)

    print("Starting model training scheduler...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass