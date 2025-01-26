import os
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from app.utils.misc import print_log, LogType

mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_URL'))

# Loading the scaler for the ML price prediction model
def load_scaler_from_mlflow(model_name):
    client = mlflow.tracking.MlflowClient()

    try:
        # Get latest model version and associated run ID
        model_version_details = client.get_latest_versions(model_name, stages=["None"])[0]
        run_id = model_version_details.run_id

        # Load the scaler from the path to scaler in artifacts
        artifact_path = "scaler/scaler.joblib"
        scaler_uri = client.download_artifacts(run_id, artifact_path)
        scaler_loaded = joblib.load(scaler_uri)

        return scaler_loaded

    except Exception as e:
        print_log("load_scaler_from_mlflow()", f"Failed loading model {model_name}: {e}", LogType.ERROR)
        return None
    
# Loading the ML price prediction model
def load_model_from_mlflow(model_name):
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    
    except Exception as e:
        print_log("load_model_from_mlflow()", f"Failed loading model {model_name}: {e}", LogType.ERROR)
        return None

# Retrieving the latest run data from MLFlow
def get_latest_run_from_mlflow(experiment_name: str, model_name: str):
    client = MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],  # Search within the specific experiment
        filter_string=f"tags.`Model name` = '{model_name}'",  # Search by model name tag
        order_by=["end_time DESC"],  # Order by end time to get the latest run
        max_results=1
    )

    if not runs:
        return None
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id

    # Retrieve details of the latest run
    run_info = client.get_run(run_id)
    run_data = run_info.data
    
    # Fetch metrics, parameters, and timestamps from the run
    metrics = run_data.metrics
    finished_time = run_info.info.end_time  # This returns the timestamp when the run ended
    
    return {"finished_time": finished_time, "mse": metrics["mse"]}