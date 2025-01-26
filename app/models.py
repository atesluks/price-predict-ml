from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Enum
from app.database import Base
from datetime import datetime, UTC
import enum

class DataCGCoinsMarketChart1h(Base):
    __tablename__ = 'data_cg_coins_market_chart_1h'

    id = Column(Integer, primary_key=True, index=True)
    pair = Column(String, nullable=False)
    time = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    mcap = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.now(UTC))

class DataCGCoinsMarketChart1d(Base):
    __tablename__ = 'data_cg_coins_market_chart_1d'

    id = Column(Integer, primary_key=True, index=True)
    pair = Column(String, nullable=False)
    time = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    mcap = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.now(UTC))

class Logs(Base):
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, default=datetime.now(UTC))
    type = Column(String, nullable=False)
    emitter = Column(String, nullable=False)
    message = Column(String, nullable=False)

class CachePrice(Base):
    __tablename__ = 'cache_price'
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.now(UTC))

class ModelTrainingStatus(enum.Enum):
    PENDING = "PENDING"
    TRAINING = "TRAINING"
    DONE = "DONE"

class ModelTrainingQueue(Base):
    __tablename__ = 'model_training_queue'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.now(UTC), nullable=False)
    updated_at = Column(DateTime, default=datetime.now(UTC), onupdate=datetime.now(UTC), nullable=False)
    data_worker = Column(String, nullable=False)
    status = Column(Enum(ModelTrainingStatus, name="model_training_status"), nullable=False)