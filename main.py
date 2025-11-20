from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os

app = FastAPI(
    title="Agricultural Commodity Price Forecasting API",
    description="API for forecasting Argentine agricultural export prices (Soy, Wheat, Corn)",
    version="1.0.0"
)

# CORS Configuration
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Agricultural Commodity Price Forecasting API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

from models import ProphetModel, LSTMForecaster
from data_loader import DataLoader
import pandas as pd

# Initialize components
data_loader = DataLoader()
# In a real app, we'd load pre-trained models. For this demo, we'll train on the fly or use cached data.
# To avoid long startup times, we can lazy load or train on first request.

@app.get("/commodities")
async def get_commodities():
    return {"commodities": list(data_loader.tickers.keys())}

@app.get("/forecast/{commodity}")
async def get_forecast(commodity: str, model_type: str = "prophet", periods: int = 30):
    """
    Get historical data and forecast for a commodity.
    model_type: 'prophet' or 'lstm'
    """
    try:
        # Fetch data
        df = data_loader.get_merged_data(commodity)
        
        forecast_result = {}
        
        if model_type == "prophet":
            model = ProphetModel()
            model.train(df)
            forecast_result = model.predict(periods=periods)
        elif model_type == "lstm":
            model = LSTMForecaster()
            model.train(df, epochs=2) # Reduced epochs for faster response on free tier
            forecast_result = model.predict(df, future_periods=periods)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
            
        # Format historical data for frontend
        historical = {
            "dates": df['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "prices": df['y'].tolist()
        }
        
        return {
            "commodity": commodity,
            "model": model_type,
            "historical": historical,
            "forecast": forecast_result
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in get_forecast: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

class ScenarioParams(BaseModel):
    temperature_change: float = 0.0 # degrees celsius change
    precipitation_change: float = 0.0 # percentage change (e.g. 0.1 for +10%)

@app.post("/scenario/{commodity}")
async def run_scenario(commodity: str, params: ScenarioParams):
    """
    Run a scenario analysis using Prophet (since it supports regressors).
    """
    try:
        df = data_loader.get_merged_data(commodity)
        
        model = ProphetModel()
        model.train(df)
        
        # Create future dataframe with adjusted weather
        future_periods = 90
        future = model.model.make_future_dataframe(periods=future_periods)
        
        # Get historical averages for weather to project future
        # This is a simplification.
        # We need to construct the future regressors dataframe
        
        # For the demo, let's just take the last year of weather and repeat it, applying the changes
        last_year_weather = df.tail(365)[['temperature', 'precipitation', 'soil_moisture']].copy()
        
        # If we need more than 365 days, we'd need to loop. For 90 days, we just take the first 90 of the last year (seasonality)
        # Actually better to take the same days from previous year to match seasonality
        
        # Let's simplify: use the last known values and apply the change constant
        # (Prophet handles seasonality of the target, but regressors need values)
        
        # Construct future regressors
        future_regressors = pd.DataFrame({
            'ds': future['ds']
        })
        
        # We need to fill 'temperature', 'precipitation', 'soil_moisture' for the future rows
        # Let's use the average of the corresponding month from history
        df['month'] = df['ds'].dt.month
        monthly_avg = df.groupby('month')[['temperature', 'precipitation', 'soil_moisture']].mean()
        
        future['month'] = future['ds'].dt.month
        future = future.merge(monthly_avg, on='month', how='left')
        
        # Apply scenario changes to the future part only
        future_mask = future['ds'] > df['ds'].max()
        
        future.loc[future_mask, 'temperature'] += params.temperature_change
        future.loc[future_mask, 'precipitation'] *= (1 + params.precipitation_change)
        
        # Predict
        forecast = model.model.predict(future)
        
        return {
            "commodity": commodity,
            "scenario": params.dict(),
            "forecast": {
                "dates": forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "predicted": forecast['yhat'].tolist(),
                "lower": forecast['yhat_lower'].tolist(),
                "upper": forecast['yhat_upper'].tolist()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

