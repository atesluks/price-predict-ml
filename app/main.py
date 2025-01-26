import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import database
from app.routers import price_prediction

# Import env variables
load_dotenv("./.env")

# Create an instance of the FastAPI class
app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    database.Base.metadata.create_all(bind=database.engine)

# Endpoints
@app.get("/")
async def read_root():
    return {"message": "Test Price Prediction ML"}

@app.get("/version")
async def read_version():
    return {"version": os.environ["VERSION"]}

# Include price prediction router
app.include_router(price_prediction.router)
