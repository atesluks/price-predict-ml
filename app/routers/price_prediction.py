from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.utils.cache_price import get_price_for_api
from app.utils.misc import get_model_configs, log

EMITTER = "price-predict-router"

model_configs = get_model_configs()

router = APIRouter()

def create_endpoint(model_name: str, lags: int, pair: str):
    async def get_price(db: Session = Depends(get_db)):
        return get_price_for_api(db, model_name, pair, lags)
    return get_price

for model in model_configs:
    endpoint_name = f"/get-{model['pair'].lower()}-price-{model['timeframe']}"
    
    # Create endpoint with fixed parameters to avoid closure issues
    endpoint_handler = create_endpoint(
        model_name=model['name'],
        lags=model['lags'], 
        pair=model['pair']
    )
    
    # Set a unique operation_id to avoid FastAPI warnings
    router.add_api_route(
        endpoint_name,
        endpoint_handler,
        methods=["GET"],
        operation_id=f"get_{model['pair'].lower()}_price_{model['timeframe']}"
    )
    log(EMITTER, f"Created endpoint {endpoint_name} for model {model['name']}")