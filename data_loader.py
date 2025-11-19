import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class DataLoader:
    def __init__(self):
        # Tickers for Soy, Wheat, Corn (using Futures)
        # ZS=F: Soybean Futures
        # ZW=F: Wheat Futures
        # ZC=F: Corn Futures
        self.tickers = {
            "soy": "ZS=F",
            "wheat": "ZW=F",
            "corn": "ZC=F"
        }
        
        # Approximate coordinates for Argentina's agricultural core (Zona Nucleo)
        self.lat = -33.0
        self.lon = -62.0

    def fetch_market_data(self, commodity: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical market data for a commodity."""
        if commodity not in self.tickers:
            raise ValueError(f"Invalid commodity. Choose from {list(self.tickers.keys())}")
        
        ticker = self.tickers[commodity]
        data = yf.download(ticker, period=period)
        
        if data.empty:
            raise ValueError(f"No data found for {commodity}")
            
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Standardize columns
        data = data[['Date', 'Close', 'Volume']]
        data.columns = ['ds', 'y', 'volume']
        
        # Ensure datetime
        data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)
        
        return data

    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data from OpenMeteo.
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_7cm_mean"],
            "timezone": "America/Sao_Paulo"
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching weather data: {response.status_code}")
            return pd.DataFrame()
            
        data = response.json()
        
        daily = data.get('daily', {})
        df = pd.DataFrame({
            'ds': daily.get('time', []),
            'temperature': daily.get('temperature_2m_mean', []),
            'precipitation': daily.get('precipitation_sum', []),
            'soil_moisture': daily.get('soil_moisture_0_to_7cm_mean', [])
        })
        
        df['ds'] = pd.to_datetime(df['ds'])
        return df

    def get_merged_data(self, commodity: str) -> pd.DataFrame:
        """Fetch and merge market and weather data."""
        market_data = self.fetch_market_data(commodity)
        
        start_date = market_data['ds'].min().strftime('%Y-%m-%d')
        end_date = market_data['ds'].max().strftime('%Y-%m-%d')
        
        weather_data = self.fetch_weather_data(start_date, end_date)
        
        if weather_data.empty:
            return market_data
            
        # Merge on date
        merged = pd.merge(market_data, weather_data, on='ds', how='left')
        
        # Fill missing weather data (forward fill)
        merged = merged.ffill()
        
        return merged
