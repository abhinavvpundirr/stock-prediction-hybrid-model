import pandas as pd
import logging
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HybridStockPredictor:
    """
    Combines LSTM and LLM predictions for stock price forecasting
    """
    def __init__(self, lstm_weight: float = 0.6, llm_weight: float = 0.4):
        """
        Initialize the hybrid predictor
        
        Args:
            lstm_weight: Weight for LSTM predictions
            llm_weight: Weight for LLM predictions
        """
        logger.info(f"Initializing Hybrid Stock Predictor with weights: LSTM={lstm_weight}, LLM={llm_weight}")
        
        # Import here to avoid circular imports
        from lstm_model import LSTMStockPredictor
        from llm_model import LLMStockPredictor
        from stock_formula import StockValueCalculator
        
        self.lstm_predictor = LSTMStockPredictor()
        self.llm_predictor = LLMStockPredictor()
        self.lstm_weight = lstm_weight
        self.llm_weight = llm_weight
        self.calculator = StockValueCalculator()
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical stock data
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        logger.info(f"Fetching historical data for {ticker} for period {period}")
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            logger.info(f"Fetched {len(data)} data points for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamentals(self, ticker: str) -> tuple:
        """
        Get fundamental data for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            tuple: (EPS, Book Value)
        """
        logger.info(f"Fetching fundamental data for {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get EPS
            eps = info.get('trailingEPS', 0)
            if eps is None or eps <= 0:
                eps = info.get('forwardEPS', 1)
            if eps is None or eps <= 0:
                eps = 1  # Default value
            
            # Get Book Value
            book_value = info.get('bookValue', 0)
            if book_value is None or book_value <= 0:
                book_value = 1  # Default value
            
            logger.info(f"Fundamentals for {ticker}: EPS={eps}, Book Value={book_value}")
            return eps, book_value
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {str(e)}")
            return 1, 1  # Default values
    
    def train(self, ticker: str) -> None:
        """
        Train the LSTM model
        
        Args:
            ticker: Stock ticker symbol
        """
        logger.info(f"Training hybrid model for {ticker}")
        data = self.get_stock_data(ticker)
        if len(data) > 0:
            self.lstm_predictor.train(data)
            logger.info(f"Hybrid model training completed for {ticker}")
        else:
            logger.error(f"No data available for {ticker}, training aborted")
    
    def predict(self, ticker: str, days_to_predict: int = 10) -> dict:
        """
        Make predictions using both LSTM and LLM models
        
        Args:
            ticker: Stock ticker symbol
            days_to_predict: Number of days to predict
            
        Returns:
            dict: Prediction results
        """
        logger.info(f"Making hybrid predictions for {ticker} for next {days_to_predict} days")
        
        # Get stock data and fundamentals
        data = self.get_stock_data(ticker)
        if len(data) == 0:
            logger.error(f"No data available for {ticker}, prediction aborted")
            return {}
        
        current_price = data['Close'].iloc[-1]
        eps, book_value = self.get_fundamentals(ticker)
        intrinsic_value = self.calculator.calculate_intrinsic_value(eps, book_value)
        
        # Get LSTM predictions
        lstm_predictions = self.lstm_predictor.predict(data, days_to_predict)
        
        # Get LLM predictions
        llm_predictions = self.llm_predictor.predict(ticker, current_price, eps, book_value, days_to_predict)
        
        # If either prediction is empty, use the other one
        if not lstm_predictions:
            logger.warning("No LSTM predictions available, using only LLM predictions")
            hybrid_predictions = llm_predictions if llm_predictions else []
        elif not llm_predictions:
            logger.warning("No LLM predictions available, using only LSTM predictions")
            hybrid_predictions = lstm_predictions
        else:
            # Combine predictions
            hybrid_predictions = []
            for i in range(min(len(lstm_predictions), len(llm_predictions))):
                hybrid_predictions.append(
                    self.lstm_weight * lstm_predictions[i] + self.llm_weight * llm_predictions[i]
                )
        
        # Find intersection points (where both models agree)
        intersection_points = []
        if lstm_predictions and llm_predictions:
            for i in range(1, min(len(lstm_predictions), len(llm_predictions))):
                # Check if the lines cross between points i-1 and i
                if ((lstm_predictions[i-1] > llm_predictions[i-1] and lstm_predictions[i] < llm_predictions[i]) or
                    (lstm_predictions[i-1] < llm_predictions[i-1] and lstm_predictions[i] > llm_predictions[i])):
                    # Calculate intersection point (linear interpolation)
                    t = ((llm_predictions[i-1] - lstm_predictions[i-1]) / 
                         ((lstm_predictions[i] - lstm_predictions[i-1]) - (llm_predictions[i] - llm_predictions[i-1])))
                    price = lstm_predictions[i-1] + t * (lstm_predictions[i] - lstm_predictions[i-1])
                    day = i-1 + t
                    intersection_points.append((day, price))
        
        # Prepare results
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_to_predict + 1)]
        
        results = {
            'ticker': ticker,
            'current_price': current_price,
            'intrinsic_value': intrinsic_value,
            'eps': eps,
            'book_value': book_value,
            'dates': dates,
            'lstm_predictions': lstm_predictions,
            'llm_predictions': llm_predictions,
            'hybrid_predictions': hybrid_predictions,
            'intersection_points': intersection_points
        }
        
        logger.info(f"Hybrid prediction completed for {ticker}")
        return results

