import pandas as pd
import numpy as np
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Optional
from datetime import timedelta

class ProphetModel:
    def __init__(self):
        self.model = None
        self.training_data = None  # Store for generating future regressors
        self.regressors = []  # Track which regressors were used

    def train(self, data: pd.DataFrame):
        """
        Train Prophet model.
        Data must have 'ds' and 'y' columns.
        Optional regressors: 'temperature', 'precipitation', 'soil_moisture'.
        """
        self.model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.training_data = data.copy()  # Store training data
        self.regressors = []  # Reset regressors list
        
        # Add regressors if they exist in data
        if 'temperature' in data.columns:
            self.model.add_regressor('temperature')
            self.regressors.append('temperature')
        if 'precipitation' in data.columns:
            self.model.add_regressor('precipitation')
            self.regressors.append('precipitation')
        if 'soil_moisture' in data.columns:
            self.model.add_regressor('soil_moisture')
            self.regressors.append('soil_moisture')
            
        self.model.fit(data)
        
        # Calculate accuracy on last 30 days as validation
        if len(data) > 30:
            train_size = len(data) - 30
            train_df = data.iloc[:train_size].copy()
            val_df = data.iloc[train_size:].copy()
            
            # Create validation model
            val_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            for reg in self.regressors:
                val_model.add_regressor(reg)
            val_model.fit(train_df)
            
            # Predict on validation set
            val_forecast = val_model.predict(val_df[['ds'] + self.regressors] if self.regressors else val_df[['ds']])
            actual = val_df['y'].values
            predicted = val_forecast['yhat'].values[:len(actual)]
            
            # Calculate MAPE and convert to accuracy percentage
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            self.accuracy = max(0, 100 - mape)  # Ensure non-negative
        else:
            self.accuracy = None

    def predict(self, periods: int = 30, future_regressors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Make predictions.
        """
        if not self.model:
            raise ValueError("Model not trained")
            
        future = self.model.make_future_dataframe(periods=periods)
        
        # If model was trained with regressors, we need to provide future values
        if self.regressors and future_regressors is None:
            # Generate future regressor values using monthly averages
            self.training_data['month'] = self.training_data['ds'].dt.month
            monthly_avg = self.training_data.groupby('month')[self.regressors].mean()
            
            future['month'] = future['ds'].dt.month
            future = future.merge(monthly_avg, on='month', how='left')
            future = future.drop('month', axis=1)
            
        elif future_regressors is not None:
            # Merge future regressors
            future = pd.merge(future, future_regressors, on='ds', how='left')
            future = future.ffill().bfill()

        forecast = self.model.predict(future)
        
        return {
            "dates": forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "predicted": forecast['yhat'].tolist(),
            "lower": forecast['yhat_lower'].tolist(),
            "upper": forecast['yhat_upper'].tolist(),
            "accuracy": round(self.accuracy, 1) if hasattr(self, 'accuracy') and self.accuracy is not None else None
        }

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class LSTMForecaster:
    def __init__(self):
        self.model = LSTM()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_window = 12

    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    def train(self, data: pd.DataFrame, epochs: int = 10):
        # Simplified training for demo
        y = data['y'].values.astype(float)
        self.scaler.fit(y.reshape(-1, 1))
        train_data_normalized = self.scaler.transform(y.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
        
        train_inout_seq = self.create_inout_sequences(train_data_normalized, self.train_window)
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                torch.zeros(1, 1, self.model.hidden_layer_size))
                
                y_pred = self.model(seq)
                
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()
        
        # Calculate accuracy on last 30 values as validation
        if len(y) > 30 + self.train_window:
            val_size = 30
            val_start = len(y) - val_size
            val_actual = y[val_start:]
            
            # Make predictions on validation set
            self.model.eval()
            val_predictions = []
            test_inputs = train_data_normalized[val_start - self.train_window:val_start].tolist()
            
            for i in range(val_size):
                seq = torch.FloatTensor(test_inputs[-self.train_window:])
                with torch.no_grad():
                    self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                    torch.zeros(1, 1, self.model.hidden_layer_size))
                    pred = self.model(seq).item()
                    test_inputs.append(pred)
                    val_predictions.append(pred)
            
            # Inverse transform predictions
            val_predictions = self.scaler.inverse_transform(np.array(val_predictions).reshape(-1, 1)).flatten()
            
            # Calculate MAPE and convert to accuracy percentage
            mape = np.mean(np.abs((val_actual - val_predictions) / val_actual)) * 100
            self.accuracy = max(0, 100 - mape)  # Ensure non-negative
        else:
            self.accuracy = None

    def predict(self, data: pd.DataFrame, future_periods: int = 30) -> Dict[str, Any]:
        # Simplified prediction
        y = data['y'].values.astype(float)
        normalized_data = self.scaler.transform(y.reshape(-1, 1))
        test_inputs = normalized_data[-self.train_window:].flatten().tolist()
        
        self.model.eval()
        
        predictions = []
        
        for i in range(future_periods):
            seq = torch.FloatTensor(test_inputs[-self.train_window:])
            with torch.no_grad():
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                torch.zeros(1, 1, self.model.hidden_layer_size))
                test_inputs.append(self.model(seq).item())
                predictions.append(test_inputs[-1])
                
        actual_predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        last_date = data['ds'].max()
        future_dates = [last_date + timedelta(days=x) for x in range(1, future_periods + 1)]
        
        return {
            "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "predicted": actual_predictions.flatten().tolist(),
            "accuracy": round(self.accuracy, 1) if hasattr(self, 'accuracy') and self.accuracy is not None else None
        }
