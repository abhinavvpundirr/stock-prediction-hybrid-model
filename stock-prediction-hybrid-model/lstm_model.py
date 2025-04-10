import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    """
    LSTM model for stock price prediction
    """
    def __init__(self, lookback_days: int = 60):
        """
        Initialize the LSTM predictor
        
        Args:
            lookback_days: Number of previous days to use for prediction
        """
        logger.info(f"Initializing LSTM Stock Predictor with lookback of {lookback_days} days")
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _create_dataset(self, data: np.ndarray) -> tuple:
        """
        Create time series dataset for LSTM
        
        Args:
            data: Stock price data
            
        Returns:
            tuple: X and y datasets
        """
        X, y = [], []
        for i in range(self.lookback_days, len(data)):
            X.append(data[i-self.lookback_days:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: tuple) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data
        """
        logger.info(f"Building LSTM model with input shape {input_shape}")
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM model built successfully")
    
    def train(self, stock_data: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train the LSTM model
        
        Args:
            stock_data: DataFrame with stock prices
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            dict: Training history
        """
        logger.info(f"Training LSTM model with {len(stock_data)} data points")
        
        # Prepare data
        data = stock_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create training dataset
        X, y = self._create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.build_model((X.shape[1], 1))
        history = self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            verbose=1
        )
        
        logger.info("LSTM model training completed")
        return history.history
    
    def predict(self, stock_data: pd.DataFrame, days_to_predict: int = 10) -> list:
        """
        Make predictions using the trained LSTM model
        
        Args:
            stock_data: DataFrame with stock prices
            days_to_predict: Number of days to predict into the future
            
        Returns:
            list: Predicted stock prices
        """
        if self.model is None:
            logger.error("Model not trained. Please train the model first.")
            return []
        
        logger.info(f"Predicting stock prices for next {days_to_predict} days")
        
        # Prepare data
        data = stock_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Create prediction dataset
        X_pred = []
        X_pred.append(scaled_data[-self.lookback_days:, 0])
        X_pred = np.array(X_pred)
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        
        # Make predictions
        predictions = []
        current_batch = X_pred
        
        for _ in range(days_to_predict):
            # Predict next value
            current_pred = self.model.predict(current_batch)[0]
            predictions.append(current_pred[0])
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        logger.info("Prediction completed")
        return predictions.flatten().tolist()

