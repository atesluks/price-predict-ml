### 0. Importing libraries and modules
import sys
import os
import time
import re

import mlflow
import logging

from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
from app.utils.model_training_queue import should_start_training, set_model_status_training, set_model_status_done
from app.ml_models.utils import load_daily_data_from_db, load_hourly_data_from_db, model_eval, retraining_whole_model, logging_results, data_pre_processing, training_model, calculate_datapoints_ahead
from app.utils.misc import get_model_configs, parse_timeframe

# Get model config
models = get_model_configs()

# Get model index from command line argument, default to 0 if not provided
model_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
model_config = models[model_index]

# Constants
ASSET_PAIR = model_config["pair"]
MODEL_NAME = model_config["name"]
MODEL_NAME = model_config["name"]
experiment_name = MODEL_NAME.replace("_", " ").title()
predict_timeframe = model_config["predict_timeframe"]
input_timeframe = model_config["input_timeframe"]
LAGS, data_type = parse_timeframe(input_timeframe)

DATAPOINTS_AHEAD_TO_PREDICT = calculate_datapoints_ahead(predict_timeframe, data_type)

TRAIN_RATIO = 0.9

OPTUNA_TRIALS = 30
RANDOM_SEARCH_N_ITER = 10
PRUNER_PATIENCE = 5

mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_URL'))
mlflow.set_experiment(experiment_name)

def scheduled_task():
    # Check if model should start training. If not, exiting.
    if not should_start_training(MODEL_NAME):
        return
    
    set_model_status_training(MODEL_NAME)
    
    # 0. Starting MLFlow run
    start_time_general = time.time()
    mlflow.start_run()

    # 1. loading the data based on data_type
    if data_type == "daily":
        df = load_daily_data_from_db(MODEL_NAME, ASSET_PAIR)
    else:  # hourly
        df = load_hourly_data_from_db(MODEL_NAME, ASSET_PAIR)

    # 2. Pre-processing the data and splitting to training and test sets
    n_train, X, y, X_train_scaled, y_train, X_test_scaled, y_test = data_pre_processing(df, MODEL_NAME, LAGS, DATAPOINTS_AHEAD_TO_PREDICT, TRAIN_RATIO)

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
        "start_time_general": start_time_general,
        "best_model_name": best_model_name,
        "predict_timeframe": predict_timeframe,
        "input_timeframe": input_timeframe,
        "lags": LAGS,
        "n_train": n_train,
        "n_test": len(y_test),
        "train_ratio": TRAIN_RATIO,
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