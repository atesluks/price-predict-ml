import math
from app.database import SessionLocal
from app.models import Logs
from datetime import datetime, UTC
import os
from enum import Enum
from dotenv import load_dotenv
import requests
import json
import yaml

load_dotenv()

SEND_SLACK_ALERTS = os.getenv("SEND_SLACK_ALERTS", "false").lower() == "true"

class LogType(Enum):
    INFO = "INFO"
    ERROR = "ERROR"
    WARN = "WARN"

def print_log(emitter = "", message = "", type = LogType.INFO):
    print(f"{datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}  [{emitter}]  [{type.value}]  {message}")

def log(emitter = "", message = "", type = LogType.INFO):
    print_log(emitter, message, type)

    logging_session = SessionLocal()
    
    log_entry = Logs(
        type=type.value,  # Use .value to get the string value
        emitter=emitter,
        message=message
    )

    logging_session.add(log_entry)
    logging_session.commit()

def log_error(emitter = "", message = ""):
    log(emitter, message, LogType.ERROR)

def log_warn(emitter = "", message = ""):
    log(emitter, message, LogType.WARN)


def is_valid_number(num) -> bool:
    return isinstance(num, (int, float)) and not math.isnan(num)

# check if an environment variable exists, raise error if not found
def get_env_var(var_name: str) -> str:
    if var_name not in os.environ:
        raise ValueError(f"Required environment variable '{var_name}' is not set")
    return os.environ[var_name]

def alert_slack(emitter, service_name, message: str) -> None:
    log_error(emitter, message)
    
    if SEND_SLACK_ALERTS:
        try:
            # Create the payload with the message
            payload = {
                "text": f"[{service_name}]  {message}"
            }
            
            # Send the POST request to the Slack webhook URL
            response = requests.post(
                get_env_var("SLACK_WEBHOOK_URL"), 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                log_error(emitter, f"Request to Slack returned an error {response.status_code}, the response is: {response.text}")

        except Exception as e:
            log_error(emitter, f"Error in alert_slack(): {e}")

# Load configs from configs.yaml file
def get_model_configs():
    try:
        # Load config file
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            
        # Validate models config
        if not config or "models" not in config:
            raise ValueError("Invalid config.yaml: 'models' section is missing")
            
        for model in config["models"]:
            # Validate required fields exist
            required_fields = ["name", "pair", "predict_timeframe", "input_timeframe"]
            for field in required_fields:
                if field not in model:
                    raise ValueError(f"Invalid config.yaml: '{field}' is missing for a model")
            
            # Validate model name is not empty or None
            if not model["name"]:
                raise ValueError("Invalid config.yaml: 'name' cannot be empty")
            
            # Validate input_timeframe format
            input_tf = model["input_timeframe"]
            if not isinstance(input_tf, str) or len(input_tf) < 2:
                raise ValueError(f"Invalid input_timeframe format: '{input_tf}'. Must be in format like '8h' or '7d'")
            
            value, unit = input_tf[:-1], input_tf[-1]
            if not value.isdigit():
                raise ValueError(f"Invalid input_timeframe value: '{value}'. Must be a number")
            if unit not in ['h', 'd']:
                raise ValueError(f"Invalid input_timeframe unit: '{unit}'. Must be 'h' (hour) or 'd' (day)")
            
            # Validate predict_timeframe format
            predict_tf = model["predict_timeframe"]
            if not isinstance(predict_tf, str) or len(predict_tf) < 2:
                raise ValueError(f"Invalid predict_timeframe format: '{predict_tf}'. Must be in format like '1h', '1d', '1w', or '1m'")
            
            value, unit = predict_tf[:-1], predict_tf[-1]
            if not value.isdigit():
                raise ValueError(f"Invalid predict_timeframe value: '{value}'. Must be a number")
            if unit not in ['h', 'd', 'w', 'm']:
                raise ValueError(f"Invalid predict_timeframe unit: '{unit}'. Must be 'h' (hour), 'd' (day), 'w' (week), or 'm' (month)")
                
        return config["models"]
        
    except FileNotFoundError:
        raise FileNotFoundError("config.yaml file not found in root directory")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config.yaml: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading config.yaml: {str(e)}")

# Extract lags and data_type from input_timeframe (e.g. "8h" -> 8 lags, "hourly" data_type)
def parse_timeframe(input_timeframe):
    lags = int(input_timeframe[:-1])  # Get all characters except last one and convert to int
    data_type = "hourly" if input_timeframe[-1] == "h" else "daily"  # "h" -> hourly, "d" -> daily
    return lags, data_type