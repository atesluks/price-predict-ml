from app.models import ModelTrainingQueue, ModelTrainingStatus
from datetime import datetime, UTC
from sqlalchemy import delete, update
from app.database import SessionLocal
from app.utils.misc import print_log
import os

PATIENCE_INTERVAL_MINUTES = int(os.getenv('PATIENCE_INTERVAL_MINUTES', 10))

HOURLY_DATA_DEPENDENT_MODELS = [
    "06_btc_1m_from_btc_8h_multi",
    "07_btc_2m_from_btc_8h_multi",
    "10_eth_1m_from_eth_8h_multi",
    "11_eth_2m_from_eth_8h_multi",
]

DAILY_DATA_DEPENDENT_MODELS = [
    "08_btc_3m_from_btc_7d_multi",
    "09_btc_6m_from_btc_7d_multi",
    "12_eth_3m_from_eth_7d_multi",
    "13_eth_6m_from_eth_7d_multi",
]

BTC_DEPENDENT_MODELS = [
    "06_btc_1m_from_btc_8h_multi",
    "07_btc_2m_from_btc_8h_multi",
    "08_btc_3m_from_btc_7d_multi",
    "09_btc_6m_from_btc_7d_multi",
]

ETH_DEPENDENT_MODELS = [
    "10_eth_1m_from_eth_8h_multi",
    "11_eth_2m_from_eth_8h_multi",
    "12_eth_3m_from_eth_7d_multi",
    "13_eth_6m_from_eth_7d_multi",
]

def add_hourly_data_dependent_models(data_worker):
    add_dependent_models(data_worker, HOURLY_DATA_DEPENDENT_MODELS)

def add_daily_data_dependent_models(data_worker):
    add_dependent_models(data_worker, DAILY_DATA_DEPENDENT_MODELS)

# Adding models to the training queue unless there are already there and are PENDING
def add_dependent_models(data_worker, dependent_models):
    session = SessionLocal()

    try:
        # Get the list of models from the database that are currently pending
        pending_models = session.query(ModelTrainingQueue.model_name).filter_by(status=ModelTrainingStatus.PENDING).all()
        
        # Extract model names from the result
        pending_model_names = [model.model_name for model in pending_models]
        
        # Iterate over the dependent models
        for model_name in dependent_models:
            if model_name not in pending_model_names:
                curr_datetime = datetime.now(UTC)
                
                # If the model is not in the list of pending models, add it to the database
                new_model_entry = ModelTrainingQueue(
                    model_name=model_name,
                    data_worker=data_worker,
                    status=ModelTrainingStatus.PENDING,
                    updated_at=curr_datetime,
                    added_at=curr_datetime
                )
                session.add(new_model_entry)

        # Commit the transaction to save the changes
        session.commit()

    finally:
        session.close()

def reset_training_status_to_pending(asset: str):
    # Determine the dependent models based on the asset
    dependent_models = BTC_DEPENDENT_MODELS if asset == "BTC" else ETH_DEPENDENT_MODELS
    
    session = SessionLocal()

    try:
        # Iterate through dependent_models and update the status from TRAINING to PENDING if it matches
        for model_name in dependent_models:
            session.execute(
                update(ModelTrainingQueue)
                .where(ModelTrainingQueue.model_name == model_name)
                .where(ModelTrainingQueue.status == ModelTrainingStatus.TRAINING)
                .values(
                    status=ModelTrainingStatus.PENDING,
                    updated_at=datetime.now(UTC)
                )
            )

            # Query to find all models with PENDING status for this model_name
            pending_models = session.query(ModelTrainingQueue).filter_by(
                model_name=model_name,
                status=ModelTrainingStatus.PENDING
            ).order_by(ModelTrainingQueue.added_at).all()

            # If more than one PENDING model is found, delete the later ones
            if len(pending_models) > 1:
                # Keep the first one and delete the rest
                for model in pending_models[1:]:
                    session.delete(model)
        
        # Commit the transaction to save the changes
        session.commit()

    finally:
        session.close()

# If the model is next in the list, it should start training.
# If the model is not the first in the list, but the first model is already training for too long, it also should start training.
# Otherwise it should not stat training.
def should_start_training(model_name: str) -> bool:
    session = SessionLocal()

    # Get all non-DONE records, ordered by id
    try:
        records = session.query(ModelTrainingQueue).filter(
            ModelTrainingQueue.status != ModelTrainingStatus.DONE
        ).order_by(ModelTrainingQueue.id).all()
    finally:
        session.close()

    # If there are no records, or the model is not in the list, return False
    if not records or model_name not in [record.model_name for record in records]:
        print_log(model_name, "Model is not in pending queue")
        return False

    first_record = records[0]

    # If the model is first in the list and is pending, it should start training
    if first_record.model_name == model_name and first_record.status == ModelTrainingStatus.PENDING:
        return True

    # If the first model is pending, the current model should not start training
    if first_record.status == ModelTrainingStatus.PENDING:
        print_log(model_name, "Other model is training")
        return False

    # Calculate time difference between the first model's `updated_at` and now
    time_diff_minutes = (datetime.now(UTC) - first_record.updated_at.replace(tzinfo=UTC)).total_seconds() / 60

    # Find the position of the model in the list
    position_in_list = next((i for i, record in enumerate(records) if record.model_name == model_name), None)

    # If the first model has been training for too long, allow the current model to start
    verdict = time_diff_minutes > position_in_list * PATIENCE_INTERVAL_MINUTES
    if not verdict:
        print_log(model_name, "Other model is training")

    return verdict

def set_model_status(model_name: str, current_status: ModelTrainingStatus, new_status: ModelTrainingStatus):
    session = SessionLocal()

    try:
        session.execute(
            update(ModelTrainingQueue)
            .where(ModelTrainingQueue.model_name == model_name)
            .where(ModelTrainingQueue.status == current_status)
            .values(
                status=new_status,
                updated_at=datetime.now(UTC)
            )
        )
        session.commit()
    finally:
        session.close()

def set_model_status_training(model_name: str):
    set_model_status(
        model_name,
        ModelTrainingStatus.PENDING,
        ModelTrainingStatus.TRAINING
    )

def set_model_status_done(model_name: str):
    set_model_status(
        model_name,
        ModelTrainingStatus.TRAINING,
        ModelTrainingStatus.DONE
    )