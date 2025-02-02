from app.models import CachePrice
from datetime import datetime, UTC, timedelta
from app.models import DataCGCoinsMarketChart1h, DataCGCoinsMarketChart1d
from app.utils.load_file import load_model_from_mlflow, load_scaler_from_mlflow
from app.utils.misc import is_valid_number, log, log_error, print_log, LogType
import pandas as pd

CACHE_REFRESH_INTERVAL_MINUTES = 5 # How often cached prices should be refreshed (in minutes)
PRICE_GUARDRAIL_MARGIN = 0.3 # How many % up and down from the current price the predicted price can go (acts as guardrails)

EMITTER = "cache_price"

# Get all cached price records from the database
def get_all_cached_price_records(session):
    cache_prices = session.query(CachePrice).all()

    cached_prices_dict = {}
    for price in cache_prices:
        cached_prices_dict[price.model_name] = {
            "price": price.price,
            "updated_at": price.updated_at
        }

    return cached_prices_dict

# Get cached price for an asset
# If no cached price or it is older than 5 min, run the ML price predict model and cache new price
def get_price(session, model_name, pair, lags, data_type="hourly"):
    try:
        # Getting cached price
        cache_record = session.query(CachePrice).filter(CachePrice.model_name == model_name).first()

        # Checking if the price is not outdated (below CACHE_REFRESH_INTERVAL_MINUTES)
        if cache_record:
            time_diff = datetime.now(UTC) - cache_record.updated_at.replace(tzinfo=UTC)
            if time_diff < timedelta(minutes=CACHE_REFRESH_INTERVAL_MINUTES) and cache_record.price:
                return cache_record.price
        
        # Running an ML price predict model
        price = predict_price(session, pair, model_name, lags, data_type)

        if not is_valid_number(price):
            log_error(EMITTER, f"Couldn't predict price with {model_name}")
            return None

        # Saving new price
        if cache_record:
            cache_record.price = price
            cache_record.updated_at = datetime.now(UTC)
            session.commit()
        else:
            new_cache_record = CachePrice(
                model_name=model_name,
                price=price,
                updated_at=datetime.now(UTC)
            )
            session.add(new_cache_record)
            session.commit()
        
        log(EMITTER, f"Cache updated for {model_name} with {price}")
        
        return price
    
    except Exception as e:
        log_error(EMITTER, f"Error in getting price for {model_name}: {e}")
        return None

def predict_price(session, pair, model_name, lags, data_type="hourly"):
    try:
        # Selecting the right table based on the data type (hourly or daily data)
        db_table = DataCGCoinsMarketChart1h if data_type == "hourly" else DataCGCoinsMarketChart1d

        # Load the data from db
        records = session.query(db_table)\
                    .filter(db_table.pair == pair)\
                    .order_by(db_table.time.desc())\
                    .limit(lags)\
                    .all()
        
        # Convert records to a DataFrame
        df = pd.DataFrame([{
            'Time': record.time,
            'Price': record.price,
            '24h_Volume': record.volume
        } for record in records])
        
        df = df.set_index('Time').sort_index() # May need check here if there is enoguh data for predictions (based on lags)
        curr_price = df.iloc[-1]['Price']

        # Create lagged features
        lagged_data = {}
        for i in range(1, len(df) + 1):
            lagged_data[f'Price_lag_{i}'] = df.iloc[-i]['Price']
            lagged_data[f'24h_Volume_lag_{i}'] = df.iloc[-i]['24h_Volume']
        lagged_df = pd.DataFrame([lagged_data])
        
        # Load the scaler
        scaler = load_scaler_from_mlflow(model_name)
        if not scaler:
            print_log(EMITTER, f"Failed loading scaler for model {model_name}", LogType.ERROR)
            return None
        
        # Scale the data
        df_scaled = scaler.transform(lagged_df)
        
        # Load the model
        model = load_model_from_mlflow(model_name)
        if not model:
            print_log(EMITTER, f"Failed loading model {model_name}", LogType.ERROR)
            return None
        
        # Make the prediction
        price = model.predict(df_scaled)
        price = float(price[0])

        min_price = curr_price * (1 - PRICE_GUARDRAIL_MARGIN)
        if price < min_price or price < 0:
            return max(min_price, 0)

        max_price = curr_price * (1 + PRICE_GUARDRAIL_MARGIN)
        if price > max_price:
            return max_price

        return price
    except Exception as e:
        print_log(EMITTER, f"Failed predicting price {model_name}: {e}", LogType.ERROR)
        return None
    
def get_price_for_api(db, model_name, pair, lags, data_type):
    price = get_price(db, model_name, pair, lags, data_type)
    if is_valid_number(price):
        return {"price": price}
    else:
        return {"error": "Something went wrong"}, 400