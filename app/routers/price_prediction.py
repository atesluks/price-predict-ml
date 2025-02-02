from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.utils.cache_price import get_price_for_api
from app.utils.misc import get_model_configs, log, parse_timeframe

EMITTER = "price-predict-router"

model_configs = get_model_configs()

router = APIRouter()

def create_endpoint(model_name: str, pair: str, lags: int, data_type: str):
    async def get_price(db: Session = Depends(get_db)):
        return get_price_for_api(db, model_name, pair, lags, data_type)
    return get_price

for model in model_configs:
    name = model['name']
    pair = model['pair']
    predict_timeframe = model["predict_timeframe"]
    input_timeframe = model["input_timeframe"]
    lags, data_type = parse_timeframe(input_timeframe)
    
    endpoint_name = f"/get-{pair.lower()}-price-in-{predict_timeframe}-from-{input_timeframe}"
    
    # Create endpoint with fixed parameters to avoid closure issues
    endpoint_handler = create_endpoint(
        model_name=name,
        pair=pair,
        lags=lags, 
        data_type=data_type
    )
    
    # Set a unique operation_id to avoid FastAPI warnings
    router.add_api_route(
        endpoint_name,
        endpoint_handler,
        methods=["GET"],
        operation_id=f"get_{pair.lower()}_price_in_{predict_timeframe}_from_{input_timeframe}"
    )
    log(EMITTER, f"Created endpoint {endpoint_name} for model {name}")