from app.models import DataCGCoinsMarketChart1h, DataCGCoinsMarketChart1d
from sqlalchemy.orm import Session
from sqlalchemy import desc

CACHE_REFRESH_INTERVAL_MINUTES = 5

COINGECKO_IDS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
}

# Retrieve the latest hourly market data records for each pair in the asset_pairs array
def get_latest_hourly_records_for_pairs(db: Session, asset_pairs: list) -> dict:
    return get_latest_records_for_pairs_impl(db, asset_pairs, DataCGCoinsMarketChart1h)

# Retrieve the latest daily market data records for each pair in the asset_pairs array
def get_latest_daily_records_for_pairs(db: Session, asset_pairs: list) -> dict:
    return get_latest_records_for_pairs_impl(db, asset_pairs, DataCGCoinsMarketChart1d)

def get_latest_records_for_pairs_impl(db: Session, asset_pairs: list, db_table) -> dict:
    latest_records = {}
    
    for pair in asset_pairs:
        # Query the latest record for the specific pair ordered by time in descending order
        latest_record = (
            db.query(db_table)
            .filter(db_table.pair == pair)
            .order_by(desc(db_table.time))
            .first()
        )

        if latest_record:
            latest_records[pair] = {
                "time": latest_record.time,
                "price": latest_record.price,
                "mcap": latest_record.mcap,
                "volume": latest_record.volume,
            }
    
    return latest_records