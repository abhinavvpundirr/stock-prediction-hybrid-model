import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import logging
import os
import sys
import json
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockValueCalculator:
    """Calculates intrinsic stock value using the formula: √(22.5 * EPS * Book Value)"""
    def calculate_intrinsic_value(self, eps, book_value):
        if eps <= 0 or book_value <= 0:
            return 0
        return math.sqrt(22.5 * eps * book_value)

class SimpleLSTMPredictor:
    """Simplified LSTM model for stock price prediction"""
    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def predict(self, stock_data, days_to_predict=10):
        # For demonstration, we'll use a simple trend-based prediction
        # In a real implementation, this would use a trained LSTM model
        data = stock_data['Close'].values
        last_price = data[-1]
        
        # Calculate average daily change over the last month
        month_data = data[-30:]
        avg_daily_change = (month_data[-1] - month_data[0]) / 30
        
        # Predict future prices based on the trend
        predictions = [last_price + avg_daily_change * i for i in range(1, days_to_predict + 1)]
        return predictions

class SimpleLLMPredictor:
    """Simplified LLM-based stock prediction"""
    def predict(self, ticker, current_price, eps, book_value, days_to_predict=10):
        # For demonstration, we'll use the intrinsic value to guide predictions
        # In a real implementation, this would use an actual LLM
        calculator = StockValueCalculator()
        intrinsic_value = calculator.calculate_intrinsic_value(eps, book_value)
        
        # If stock is undervalued, predict gradual increase
        if intrinsic_value > current_price:
            step = (intrinsic_value - current_price) / (days_to_predict * 2)
            predictions = [current_price + step * i for i in range(1, days_to_predict + 1)]
        # If stock is overvalued, predict gradual decrease
        else:
            step = (current_price - intrinsic_value) / (days_to_predict * 2)
            predictions = [current_price - step * i for i in range(1, days_to_predict + 1)]
            
        return predictions

class HybridPredictor:
    """Combines LSTM and LLM predictions"""
    def __init__(self, lstm_weight=0.6, llm_weight=0.4):
        self.lstm_predictor = SimpleLSTMPredictor()
        self.llm_predictor = SimpleLLMPredictor()
        self.lstm_weight = lstm_weight
        self.llm_weight = llm_weight
        self.calculator = StockValueCalculator()
    
    def get_stock_data(self, ticker, period="1y"):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get EPS
            eps = info.get('trailingEPS', 0)
            if eps is None or eps <= 0:
                eps = info.get('forwardEPS', 1)
            if eps is None or eps <= 0:
                eps = 1
            
            # Get Book Value
            book_value = info.get('bookValue', 0)
            if book_value is None or book_value <= 0:
                book_value = 1
            
            return eps, book_value
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {str(e)}")
            return 1, 1
    
    def predict(self, ticker, days_to_predict=10):
        # Get stock data and fundamentals
        data = self.get_stock_data(ticker)
        if len(data) == 0:
            return {}
        
        current_price = data['Close'].iloc[-1]
        eps, book_value = self.get_fundamentals(ticker)
        intrinsic_value = self.calculator.calculate_intrinsic_value(eps, book_value)
        
        # Get predictions
        lstm_predictions = self.lstm_predictor.predict(data, days_to_predict)
        llm_predictions = self.llm_predictor.predict(ticker, current_price, eps, book_value, days_to_predict)
        
        # Combine predictions
        hybrid_predictions = []
        for i in range(min(len(lstm_predictions), len(llm_predictions))):
            hybrid_predictions.append(
                self.lstm_weight * lstm_predictions[i] + self.llm_weight * llm_predictions[i]
            )
        
        # Find intersection points
        intersection_points = []
        for i in range(1, min(len(lstm_predictions), len(llm_predictions))):
            if ((lstm_predictions[i-1] > llm_predictions[i-1] and lstm_predictions[i] < llm_predictions[i]) or
                (lstm_predictions[i-1] < llm_predictions[i-1] and lstm_predictions[i] > llm_predictions[i])):
                # Calculate intersection point
                t = ((llm_predictions[i-1] - lstm_predictions[i-1]) / 
                     ((lstm_predictions[i] - lstm_predictions[i-1]) - (llm_predictions[i] - llm_predictions[i-1])))
                price = lstm_predictions[i-1] + t * (lstm_predictions[i] - lstm_predictions[i-1])
                day = i-1 + t
                intersection_points.append((day, price))
        
        # Prepare dates
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_to_predict + 1)]
        
        # Prepare results
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
        
        return results

def plot_predictions(results):
    """Plot the prediction results"""
    if not results:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    dates = results['dates']
    plt.plot(dates, results['lstm_predictions'], label='LSTM Predictions', marker='o')
    plt.plot(dates, results['llm_predictions'], label='LLM Predictions', marker='s')
    plt.plot(dates, results['hybrid_predictions'], label='Hybrid Predictions', marker='^')
    
    # Plot intersection points
    for day, price in results['intersection_points']:
        plt.scatter([dates[int(day)]], [price], color='red', s=100, zorder=5, 
                   label='Intersection Point' if 'Intersection Point' not in plt.gca().get_legend_handles_labels()[1] else '')
    
    # Plot current price and intrinsic value
    plt.axhline(y=results['current_price'], color='gray', linestyle='--', label=f"Current Price (${results['current_price']:.2f})")
    plt.axhline(y=results['intrinsic_value'], color='green', linestyle='--', label=f"Intrinsic Value (${results['intrinsic_value']:.2f})")
    
    plt.title(f"Stock Price Prediction for {results['ticker']}")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{results['ticker']}_prediction.png")
    plt.show()

def main():
    """Run predictions for a user-specified ticker"""
    print("=" * 50)
    print("Stock Price Prediction with Hybrid ML/LLM Approach")
    print("=" * 50)
    
    # Get user input for ticker
    ticker = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT, GOOGL): ").strip().upper()
    
    # Validate ticker
    if not ticker:
        print("Error: Ticker symbol cannot be empty.")
        return
    
    # Get user input for prediction days
    try:
        days_input = input("Enter number of days to predict (default: 10): ").strip()
        days_to_predict = int(days_input) if days_input else 10
    except ValueError:
        print("Invalid input. Using default value of 10 days.")
        days_to_predict = 10
    
    # Get user input for model weights
    try:
        lstm_input = input("Enter LSTM weight (0-1, default: 0.6): ").strip()
        lstm_weight = float(lstm_input) if lstm_input else 0.6
        
        llm_input = input("Enter LLM weight (0-1, default: 0.4): ").strip()
        llm_weight = float(llm_input) if llm_input else 0.4
        
        # Validate weights
        if not (0 <= lstm_weight <= 1) or not (0 <= llm_weight <= 1):
            print("Weights must be between 0 and 1. Using default values.")
            lstm_weight, llm_weight = 0.6, 0.4
    except ValueError:
        print("Invalid input. Using default weights.")
        lstm_weight, llm_weight = 0.6, 0.4
    
    print(f"\nProcessing {ticker} with {days_to_predict} days prediction horizon...")
    print(f"Model weights: LSTM={lstm_weight}, LLM={llm_weight}")
    print("-" * 50)
    
    # Initialize predictor with user-specified weights
    predictor = HybridPredictor(lstm_weight=lstm_weight, llm_weight=llm_weight)
    
    try:
        # Make predictions
        results = predictor.predict(ticker, days_to_predict)
        
        if not results:
            print(f"Error: Could not retrieve data for {ticker}. Please check the ticker symbol.")
            return
        
        # Print summary
        print(f"\nSummary for {ticker}:")
        print(f"Current Price: ${results['current_price']:.2f}")
        print(f"Intrinsic Value: ${results['intrinsic_value']:.2f}")
        print(f"EPS: ${results['eps']:.2f}")
        print(f"Book Value: ${results['book_value']:.2f}")
        
        # Valuation assessment
        if results['intrinsic_value'] > results['current_price'] * 1.2:
            print("Valuation: Significantly Undervalued ⭐⭐⭐")
        elif results['intrinsic_value'] > results['current_price'] * 1.05:
            print("Valuation: Slightly Undervalued ⭐⭐")
        elif results['intrinsic_value'] < results['current_price'] * 0.8:
            print("Valuation: Significantly Overvalued ⚠️⚠️⚠️")
        elif results['intrinsic_value'] < results['current_price'] * 0.95:
            print("Valuation: Slightly Overvalued ⚠️")
        else:
            print("Valuation: Fairly Valued ⭐")
        
        print("\nPredictions for next", days_to_predict, "days:")
        print("-" * 70)
        print(f"{'Date':<12} | {'LSTM':<10} | {'LLM':<10} | {'Hybrid':<10}")
        print("-" * 70)
        
        for i, date in enumerate(results['dates']):
            print(f"{date:<12} | ${results['lstm_predictions'][i]:<9.2f} | ${results['llm_predictions'][i]:<9.2f} | ${results['hybrid_predictions'][i]:<9.2f}")
        
        if results['intersection_points']:
            print("\nIntersection Points (where models agree):")
            for day, price in results['intersection_points']:
                date_idx = int(day)
                date = results['dates'][date_idx] if date_idx < len(results['dates']) else "Future"
                print(f"Day {day+1:.1f} ({date}): ${price:.2f}")
        
        # Plot predictions
        print("\nGenerating prediction plot...")
        plot_predictions(results)
        print(f"Plot saved as {ticker}_prediction.png")
        
        # Save results to JSON
        with open(f"{ticker}_prediction.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {ticker}_prediction.json")
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()