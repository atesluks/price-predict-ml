import os
import time
import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
import mlflow
from mlflow.models import infer_signature
import optuna

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, QuantileRegressor,
    RANSACRegressor, TheilSenRegressor, PoissonRegressor, GammaRegressor, TweedieRegressor
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.database import SessionLocal
from app.models import DataCGCoinsMarketChart1h, DataCGCoinsMarketChart1d
from sklearn.model_selection import TimeSeriesSplit

# Constants
CROSS_VAL_SPLIT = 5
OPTUNA_TRIALS = 30
RANDOM_SEARCH_N_ITER = 10
PRUNER_PATIENCE = 5

# Functions

def load_hourly_data_from_db(model_name, pair):
    return load_data_from_db(model_name, pair, DataCGCoinsMarketChart1h)

def load_daily_data_from_db(model_name, pair):
    return load_data_from_db(model_name, pair, DataCGCoinsMarketChart1d)

def load_data_from_db(model_name, pair, db_table):
    session = SessionLocal()
    try:
        data = session.query(db_table)\
            .filter(db_table.pair == pair)\
            .order_by(db_table.time.desc())\
            .all()
        
        print(f"{model_name}: Data loaded.")
        
        df = pd.DataFrame([{
            'Time': record.time,
            'Pair': record.pair,
            'Price': record.price,
            '24h_Volume': record.volume
        } for record in data])
        
        df.set_index('Time', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        print(f"{model_name}: Total records values: {len(df)}")

        # Dropping Pair column
        return df.drop(columns='Pair')
    finally:
        session.close()

def scale(X):
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the data
    X_scaled = scaler.fit_transform(X)

    return scaler, X_scaled

def retraining_whole_model(fresh_model, X, y):
    # Scale data
    scaler, X_scaled = scale(X)

    # Train model on the whole scaled dataset
    model = fresh_model.fit(X_scaled, y)

    return scaler, model

def parse_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    minutes = int(minutes)
    seconds = round(seconds)

    if minutes == 0:
        return f"{seconds}s"
    else:
        return f"{int(minutes)}m {seconds}s"

# Pre-processing the data
def data_pre_processing(df, model_name, lags, time_ahead_to_predict, n_test):
    # Checking missing values
    print(f"{model_name}: Total missing values - {df.isna().sum().sum()}")

    # Create lagged features for the last 12 hours
    def create_lagged_features(df):
        prediction_price_lag = time_ahead_to_predict - 1

        for lag in range(1, lags + 1):
            df[f'Price_lag_{lag}'] = df['Price'].shift(lag)
            df[f'24h_Volume_lag_{lag}'] = df['24h_Volume'].shift(lag)

        df[f'Price_in_{time_ahead_to_predict}'] = df['Price'].shift(-prediction_price_lag)
        
        # Drop rows with any NaN values created by the lagging process
        df.dropna(inplace=True)
        return df

    df_lagged = create_lagged_features(df)
    df_lagged = df_lagged.drop(columns="24h_Volume")
    df_lagged = df_lagged.drop(columns="Price")

    # Checking total and missing values
    print(f"{model_name}: Total records with lagged values: {len(df_lagged)}")
    print(f"{model_name}: Total missing values - {df_lagged.isna().sum().sum()}")

    # Split the data to training, validation and testing datasets
    df_lagged = df_lagged.sort_index(ascending=True)
    X = df_lagged.drop(columns=f'Price_in_{time_ahead_to_predict}')
    y = df_lagged[f'Price_in_{time_ahead_to_predict}']

    # Split the data into training and testing datasets
    n_train = len(df_lagged) - n_test

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]

    print(f"{model_name}: Shapes of X_train, y_train, X_test, y_test: {X_train.shape, y_train.shape, X_test.shape, y_test.shape}")

    # Fit and scale on the training data
    scaler, X_train_scaled = scale(X_train)

    # Scale the validation data
    X_test_scaled = scaler.transform(X_test)

    return n_train, X, y, X_train_scaled, y_train, X_test_scaled, y_test
    
# Predicting and Evaluating the model
def model_eval(model_name, model, X_test_scaled, y_test):
    # Predicting
    y_pred = model.predict(X_test_scaled)

    # Calculating performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE as a percentage
    r2 = model.score(X_test_scaled, y_test)

    # Print results
    print(f"{model_name}: MSE - {mse:.8f}; MAE - {mae:.8f}; RMSE - {rmse:.8f}; MAPE - {mape:.2f}%; R2 - {r2 * 100:.2f}%")

    return mse, mae, rmse, mape, r2, y_pred

# Logging to MLFlow
def logging_results(
        params,
        metrics,
        scaler,
        model,
        X_train_scaled, 
        y_pred,
        random_search_all_model_variations_by_score, 
        random_search_all_models_by_time,
        random_search_top_10_models_by_score, 
        optimization_results_df,
    ):
    model_name = params["model_name"]
    best_model_name = params["best_model_name"]

    # Save the scaler to a file
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists(f'data/{model_name}'):
        os.makedirs(f'data/{model_name}')

    scaler_path = f"data/{model_name}/scaler.joblib"
    joblib.dump(scaler, scaler_path)

    end_time = time.time()
    total_time = parse_time(end_time - params["start_time_general"])

    # Log the scaler as an additional artifact
    mlflow.log_artifact(scaler_path, artifact_path="scaler")

    # Set tags
    mlflow.set_tag("Model name", model_name)
    mlflow.set_tag("mlflow.note.content", params["model_description"])
    mlflow.set_tag("Model type", best_model_name)
    mlflow.set_tag("Lags", params["lags"])
    mlflow.set_tag("Input description", params["input_description"])
    mlflow.set_tag("Output description", params["output_description"])
    mlflow.set_tag("Train data points", params["n_train"])
    mlflow.set_tag("Test data points", params["n_test"])
    mlflow.set_tag("Source data name", params["source_name"])
    mlflow.set_tag("Source data columns", params["source_columns"])
    mlflow.set_tag("Source data time form", params["data_from"])
    mlflow.set_tag("Source data time to", params["data_to"])

    # Logging time
    mlflow.set_tag("Random Search total time", params["random_search_total_time"])
    mlflow.set_tag("Random Search top10 moels time", params["random_search_top10_total_time"])
    mlflow.set_tag("Bayesian optimization time", params["bayes_opt_total_time"])
    mlflow.set_tag("Total time", total_time)

    # Save intermediate results
    tmp_dict = random_search_all_model_variations_by_score.to_dict(orient="records")
    mlflow.log_dict(tmp_dict, "results/random_search_all_model_variations_by_score.json")

    tmp_dict = random_search_all_models_by_time.to_dict(orient="records")
    mlflow.log_dict(tmp_dict, "results/random_search_all_models_by_time.json")

    tmp_dict = random_search_top_10_models_by_score.to_dict(orient="records")
    mlflow.log_dict(tmp_dict, "results/random_search_top_10_models_by_score.json")

    tmp_dict = optimization_results_df.to_dict(orient="records")
    mlflow.log_dict(tmp_dict, "results/optimization_results_df.json")
    
    # Log parameters
    model_params = model.get_params()
    mlflow.log_params(model_params)
    
    # Log metrics
    mlflow.log_metric("mse", metrics["mse"])
    mlflow.log_metric("mae", metrics["mae"])
    mlflow.log_metric("rmse", metrics["rmse"])
    mlflow.log_metric("mape", metrics["mape"])
    mlflow.log_metric("r2", metrics["r2"])
    
    # Log the model
    signature = infer_signature(X_train_scaled, y_pred)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=best_model_name,
        signature=signature,
        input_example=X_train_scaled,
        registered_model_name=model_name,
    )

    # Removing the temp scaler file
    if os.path.exists(scaler_path):
        os.remove(scaler_path)
        
    # Print the run ID
    print(f"Logged data and model in run {mlflow.active_run().info.run_id}")

def training_model(X_train_scaled, y_train, X_test_scaled, y_test):
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {}
        },
        "RidgeRegression": {
            "model": Ridge(),
            "params": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 15.0, 30.0, 50.0, 70.0, 100.0, 200.0]
            }
        },
        "LassoRegression": {
            "model": Lasso(),
            "params": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 15.0, 30.0, 50.0, 70.0, 100.0, 200.0]
            }
        },
        "ElasticNet": {
            "model": ElasticNet(),
            "params": {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            }
        },
        "BayesianRidge": {
            "model": BayesianRidge(),
            "params": {
                'max_iter': [100, 300, 500, 1000],
                'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
                'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
                'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
                'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3]
            }
        },
        "ARDRegression": {
            "model": ARDRegression(),
            "params": {
                'max_iter': [100, 300, 500, 1000],
                'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
                'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
                'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
                'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3]
            }
        },
        "SGDRegressor": {
            "model": SGDRegressor(),
            "params": {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': [0.01, 0.1, 1.0]
            }
        },
        "PassiveAggressiveRegressor": {
            "model": PassiveAggressiveRegressor(),
            "params": {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 1.0]
            }
        },
        "HuberRegressor": {
            "model": HuberRegressor(),
            "params": {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'epsilon': [1.35, 1.5, 1.75, 2.0]
            }
        },
        "QuantileRegressor": {
            "model": QuantileRegressor(),
            "params": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'quantile': [0.1, 0.25, 0.5, 0.75, 0.9]
            }
        },
        "RANSACRegressor": {
            "model": RANSACRegressor(),
            "params": {
                'min_samples': [0.1, 0.25, 0.5, 0.75, 1.0],
                'residual_threshold': [None, 1.0, 2.0, 5.0, 10.0],
                'max_trials': [100, 500, 1000]
            }
        },
        # This is the best model, but run too long in random search (e.g. 16 minutes, while other run max 1 min, but mostly below that)
        # "TheilSenRegressor": {
        #     "model": TheilSenRegressor(),
        #     "params": {
        #         'max_subpopulation': [1e4, 1e5, 1e6],
        #         'n_subsamples': [None, 300, 500],
        #         'max_iter': [300, 500],
        #         'tol': [1e-3, 1e-4]
        #     }
        # },
        "PoissonRegressor": {
            "model": PoissonRegressor(),
            "params": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'max_iter': [100, 300, 500, 1000]
            }
        },
        "GammaRegressor": {
            "model": GammaRegressor(),
            "params": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'max_iter': [100, 300, 500, 1000]
            }
        },
        "TweedieRegressor": {
            "model": TweedieRegressor(),
            "params": {
                'power': [0, 1, 1.5, 2, 3],
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [100, 300, 500, 1000]
            }
        },
        # Commenting out this model for now because there are issues with pipeline
        # (when using .__class__ it returns a pipeline, therefore cannot just initilize
        # a fresh model from it
        # "PolynomialRegression": {
        #     "model": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        #     "params": {}
        # },
        "SVR": {
            "model": SVR(),
            "params": {
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']  # Only for 'rbf', 'poly', and 'sigmoid' kernels
            }
        },
        "DecisionTreeRegressor": {
            "model": DecisionTreeRegressor(),
            "params": {
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 10]
            }
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(),
            "params": {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 10],
                'bootstrap': [True, False]
            }
        },
        "ExtraTreesRegressor": {
            "model": ExtraTreesRegressor(),
            "params": {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 10],
                'bootstrap': [True, False]
            }
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor(),
            "params": {
                'n_estimators': [50, 100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 10]
            }
        },
        "AdaBoostRegressor": {
            "model": AdaBoostRegressor(),
            "params": {
                'n_estimators': [50, 100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'loss': ['linear', 'square', 'exponential']
            }
        },
        "HistGradientBoostingRegressor": {
            "model": HistGradientBoostingRegressor(),
            "params": {
                'max_iter': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [None, 10, 20, 30],
                'min_samples_leaf': [10, 20, 30],
                'l2_regularization': [0.0, 0.1, 1.0]
            }
        },
        "KNeighborsRegressor": {
            "model": KNeighborsRegressor(),
            "params": {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20, 30, 40]
            }
        },
        # need more testing and optimization, it throws many different errors and warning because of different kernels
        # "GaussianProcessRegressor": {
        #     "model": GaussianProcessRegressor(),
        #     "params": {
        #         'alpha': [1e-10, 1e-5, 1e-2, 0.1, 1.0, 10.0, 100.0],
        #         'kernel': [RBF(length_scale_bounds=(1e-2, 1e5)),
        #                    Matern(length_scale_bounds=(1e-2, 1e5)),
        #                    RationalQuadratic(length_scale_bounds=(1e-2, 1e5)),
        #                    ExpSineSquared(length_scale_bounds=(1e-2, 1e5))],
        #         'n_restarts_optimizer': [0, 1, 2, 5]
        #     }
        # },
        "MLPRegressor": {
            "model": MLPRegressor(),
            "params": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100)],
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'max_iter': [200, 300, 500]
            }
        }
    }

    # Store the results
    results = []
    random_search_start_time = time.time()

    # Training data being split for window cross-validation (specific for time series type of data)
    tscv = TimeSeriesSplit(n_splits=CROSS_VAL_SPLIT)

    # Train and test each model with random search
    for model_name, model_dict in models.items():
        model = model_dict["model"]
        param_distributions = model_dict["params"]

        start_time = time.time()
        
        if param_distributions:
            # Perform random search if there are hyperparameters to tune
            random_search = RandomizedSearchCV(
                model, 
                param_distributions, 
                n_iter=RANDOM_SEARCH_N_ITER, 
                cv=tscv,
                scoring='neg_mean_squared_error', 
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(X_train_scaled, y_train)
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            score = -random_search.best_score_

            end_time = time.time()
            time_diff = end_time - start_time
            
            # Collecting results for all tried hyperparameters
            for i in range(len(random_search.cv_results_['mean_test_score'])):
                results.append({
                    'model': model_name,
                    'params': random_search.cv_results_['params'][i],
                    'mse_score': -random_search.cv_results_['mean_test_score'][i],
                    'time_spent_seconds': time_diff
                })

            print(f"{model_name} - Best Params: {best_params} - MSE: {score:.10f} - Time Spent: {parse_time(time_diff)} seconds")
        else:
            # Manually fit the model with sliding window cross-validation
            score_results = []
            best_model = model
            for train_index, test_index in tscv.split(X_train_scaled):
                X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
                y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
                
                best_model.fit(X_train_fold, y_train_fold)

                # Predict on the test fold and collect results
                y_pred_fold = best_model.predict(X_test_fold)
                score_fold = mean_squared_error(y_test_fold, y_pred_fold)

                score_results.append(score_fold)
            
            # Record result for the current fold
            end_time = time.time()
            time_diff = end_time - start_time
            score = sum(score_results) / len(score_results)
            results.append({
                'model': model_name,
                'params': {},  # No hyperparameters in this case
                'mse_score': score, # average from the scores from each cross-validation folds
                'time_spent_seconds': time_diff,
            })

            print(f"{model_name} - MSE: {score:.10f} - Time Spent: {parse_time(time_diff)} seconds")

    # Recording time spent on Random Search
    random_search_end_time = time.time()
    random_search_total_time = parse_time(random_search_end_time - random_search_start_time)
    print(f"Random Search total time: {random_search_total_time}")

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)
    results_df['time_spent'] = results_df['time_spent_seconds'].apply(parse_time)

    # Just for logging - Calculate time spent on Random Search for each model
    random_search_all_models_by_time = results_df.drop_duplicates('model').sort_values(by='time_spent_seconds', ascending=False)

    # Select the top 10 models by MSE score, ensuring no duplicate model types
    random_search_all_model_variations_by_score = results_df.sort_values(by='mse_score')
    random_search_top_10_models_by_score = random_search_all_model_variations_by_score.drop_duplicates('model').head(10)
    random_search_top10_total_time = parse_time(random_search_top_10_models_by_score["time_spent_seconds"].sum())
    print(f"Random search top10 time: {random_search_top10_total_time}")

    # Define a function to perform Optuna optimization for a given model and its hyperparameters
    def objective(trial, model_class, param_distributions, best_params):
        model_params = {}

        for param, value in best_params.items():
            if isinstance(value, bool): # boolean is a subclass of int, so this check should come first
                model_params[param] = trial.suggest_categorical(param, param_distributions[param])
            elif value is None:
                # Treat None as a categorical option
                model_params[param] = trial.suggest_categorical(param, [None] + param_distributions[param])
            elif isinstance(value, (int, float)):
                max_bound = max(v for v in param_distributions[param] if v is not None)
                min_bound = min(v for v in param_distributions[param] if v is not None)

                # Searching in the area of the best_value (found in the random search) +/-10% of the whole range
                range_frac = (max_bound - min_bound) * 0.1
                
                if isinstance(value, int):
                    # Ensure lower bound is not below min value in param distribution
                    lower_bound = max(min_bound, int(value - range_frac)) 
                    
                    # Ensure upper bound is not above max value in param distribution, and is higher than lower bound
                    upper_bound = min(max_bound, max(lower_bound + 1, int(value + range_frac)))
                    
                    model_params[param] = trial.suggest_int(param, lower_bound, upper_bound)
                elif isinstance(value, float):
                    # Ensure lower bound is not below min value in param distribution
                    lower_bound = max(min_bound, value - range_frac) 
                    
                    # Ensure upper bound is not above max value in param distribution, and is higher than lower bound
                    upper_bound = min(max_bound, max(lower_bound + 0.01, value + range_frac))
                        
                    model_params[param] = trial.suggest_float(param, lower_bound, upper_bound)
            else:
                model_params[param] = trial.suggest_categorical(param, param_distributions[param])

        model = model_class(**model_params)
        
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = mean_squared_error(y_test, y_pred)
        except np.linalg.LinAlgError as e:
            # If a LinAlgError occurs, assign a high error value to penalize this trial
            score = float('inf')
        return score

    # Initialize a list to store results
    optimization_results = []
    bayes_opt_start_time = time.time()

    # Perform Bayesian optimization for each of the top 10 models
    def optimize_model(row):
        model_name = row['model']
        model_class = models[model_name]["model"].__class__
        param_distributions = models[model_name]["params"]
        
        start_time = time.time()

        if param_distributions:
            best_params = row['params']
            pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=PRUNER_PATIENCE)
            study = optuna.create_study(direction='minimize', pruner=pruner)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(lambda trial: objective(trial, model_class, param_distributions, best_params), n_trials=OPTUNA_TRIALS)
            
            # Store the results
            best_trial_params = study.best_trial.params
            score = study.best_trial.value
            time_diff = time.time() - start_time
            
            result = {
                'model_name': model_name,
                'best_params': best_trial_params,
                'best_score': score,
                'best_model': model_class(**best_trial_params),
                'time_spent_seconds': time_diff
            }

            print(f"{model_name}: Best params - {best_trial_params}, MSE - {score:.10f}, time - {parse_time(time_diff)} seconds")
        else:
            # Directly fit the model if no hyperparameters to tune
            score = row['mse_score']
            time_diff = time.time() - start_time

            # Store the results
            result = {
                'model_name': model_name,
                'best_params': {},
                'best_score': score,
                'best_model': model_class(),
                'time_spent_seconds': time_diff
            }

            print(f"{model_name}: MSE - {score:.10f}, time - {parse_time(time_diff)} seconds")

        return result
    
    # Calculating time spent on Bayesian Optimization
    optimization_results = Parallel(n_jobs=-1)(delayed(optimize_model)(row) for _, row in random_search_top_10_models_by_score.iterrows())
    bayes_opt_end_time = time.time()
    bayes_opt_total_time = parse_time(bayes_opt_end_time - bayes_opt_start_time)
    print("Optimization complete.")
    print(f"Bayesian optimization total time: {bayes_opt_total_time}")

    # Convert optimization results to a DataFrame for better readability and sorting
    optimization_results_df = pd.DataFrame([{
        'model_name': result['model_name'],
        'best_params': result['best_params'],
        'best_score': result['best_score'],
        'time_spent_seconds': result['time_spent_seconds']
    } for result in optimization_results])

    optimization_results_df['time_spent'] = optimization_results_df['time_spent_seconds'].apply(parse_time)

    # Sort the results by the best_score (MSE)
    optimization_results_df = optimization_results_df.sort_values(by='best_score').reset_index(drop=True)

    # Selecting the best model from Bayesian optimization
    best_model_name = optimization_results_df.iloc[0]['model_name']

    # getting the best model
    for result in optimization_results:
        if result['model_name'] == best_model_name:
            best_model = result['best_model'].fit(X_train_scaled, y_train)
            fresh_best_model = result['best_model'].__class__(**result['best_params'])
            return (
                best_model_name, 
                best_model, 
                fresh_best_model, 
                random_search_all_model_variations_by_score, 
                random_search_all_models_by_time,
                random_search_top_10_models_by_score, 
                optimization_results_df,
                random_search_total_time,
                random_search_top10_total_time,
                bayes_opt_total_time
            )

    # Taking the best model
    print("ERROR: Best model was not returned.")
    return (
        best_model_name, 
        None, 
        None, 
        random_search_all_model_variations_by_score, 
        random_search_all_models_by_time,
        random_search_top_10_models_by_score, 
        optimization_results_df,
        random_search_total_time,
        random_search_top10_total_time,
        bayes_opt_total_time
    )