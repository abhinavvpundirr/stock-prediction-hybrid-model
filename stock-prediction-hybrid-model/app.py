import gradio as gr
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
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import modules
from logger import setup_logger
from stock_formula import StockValueCalculator
from lstm_model import LSTMStockPredictor
from llm_model import LLMStockPredictor
from hybrid_model import HybridStockPredictor

# Setup logger
logger = setup_logger("app")

# Initialize the hybrid predictor
predictor = HybridStockPredictor()

def get_stock_info(ticker):
    """Get basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price
        current_price = info.get('currentPrice', 0)
        if current_price is None or current_price == 0:
            history = stock.history(period="1d")
            if not history.empty:
                current_price = history['Close'].iloc[-1]
            else:
                current_price = 0
        
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
        
        # Get company name
        name = info.get('longName', ticker)
        
        # Get sector
        sector = info.get('sector', 'Unknown')
        
        # Get industry
        industry = info.get('industry', 'Unknown')
        
        return {
            'name': name,
            'ticker': ticker,
            'current_price': current_price,
            'eps': eps,
            'book_value': book_value,
            'sector': sector,
            'industry': industry
        }
    except Exception as e:
        logger.error(f"Error getting stock info for {ticker}: {str(e)}")
        return {
            'name': ticker,
            'ticker': ticker,
            'current_price': 0,
            'eps': 1,
            'book_value': 1,
            'sector': 'Unknown',
            'industry': 'Unknown'
        }

def get_historical_prices(ticker, days=10):
    """Get historical prices for the past N days"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+10)  # Get a bit more data than needed
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        
        if history.empty:
            logger.warning(f"No historical data found for {ticker}")
            return [stock.info.get('currentPrice', 0)] * days
        
        # Get the last N days of closing prices
        closing_prices = history['Close'].tail(days).tolist()
        
        # If we don't have enough data, pad with the last known price
        if len(closing_prices) < days:
            last_price = closing_prices[-1] if closing_prices else stock.info.get('currentPrice', 0)
            closing_prices = closing_prices + [last_price] * (days - len(closing_prices))
            
        return closing_prices
    except Exception as e:
        logger.error(f"Error getting historical prices for {ticker}: {str(e)}")
        return [0] * days

def create_prediction_plot(results):
    """Create a plot of the prediction results"""
    fig = plt.figure(figsize=(10, 6))
    
    # Plot predictions
    dates = results['dates']
    plt.plot(dates, results['lstm_predictions'], label='LSTM Predictions', marker='o')
    plt.plot(dates, results['llm_predictions'], label='LLM Predictions', marker='s')
    plt.plot(dates, results['hybrid_predictions'], label='Hybrid Predictions', marker='^')
    
    # Plot intersection points
    for day, price in results['intersection_points']:
        plt.scatter([dates[int(day)]], [price], color='red', s=100, zorder=5, 
                   label='Intersection Point' if 'Intersection Point' not in plt.gca().get_legend_handles_labels()[1] else '')
    
    # Plot current/historical prices instead of a single horizontal line
    if 'historical_prices' in results and len(results['historical_prices']) == len(dates):
        plt.plot(dates, results['historical_prices'], color='gray', linestyle='--', 
                label=f"Historical/Current Prices", marker='x')
    else:
        # Fallback to horizontal line if historical data isn't available
        plt.axhline(y=results['current_price'], color='gray', linestyle='--', 
                  label=f"Current Price (${results['current_price']:.2f})")
    
    # Plot intrinsic value
    plt.axhline(y=results['intrinsic_value'], color='green', linestyle='--', 
               label=f"Intrinsic Value (${results['intrinsic_value']:.2f})")
    
    plt.title(f"Stock Price Prediction for {results['ticker']}")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return fig

def predict_stock(ticker, days_to_predict=10, lstm_weight=0.6, llm_weight=0.4):
    """Make predictions for a stock"""
    logger.info(f"Making predictions for {ticker}")
    
    try:
        # Get stock info
        stock_info = get_stock_info(ticker)
        
        # Update weights
        predictor.lstm_weight = lstm_weight
        predictor.llm_weight = llm_weight
        
        # Get historical prices for the past N days
        historical_prices = get_historical_prices(ticker, days_to_predict)
        
        # Get historical dates for the past N days (instead of future dates)
        end_date = datetime.now()
        dates = []
        for i in range(days_to_predict-1, -1, -1):
            date = end_date - timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))
        
        # Train the model
        predictor.train(ticker)
        
        # Make predictions
        results = predictor.predict(ticker, days_to_predict)
        
        if not results:
            return None, f"Failed to make predictions for {ticker}", None
        
        # Override the dates in results with historical dates
        results['dates'] = dates
        
        # Add historical prices to results
        results['historical_prices'] = historical_prices
        
        # Create plot
        fig = create_prediction_plot(results)
        
        # Create summary
        summary = f"## Stock Analysis for {stock_info['name']} ({ticker})\n\n"
        summary += f"**Sector:** {stock_info['sector']}\n\n"
        summary += f"**Industry:** {stock_info['industry']}\n\n"
        summary += f"**Current Price:** ${results['current_price']:.2f}\n\n"
        summary += f"**Intrinsic Value:** ${results['intrinsic_value']:.2f}\n\n"
        summary += f"**EPS:** ${results['eps']:.2f}\n\n"
        summary += f"**Book Value:** ${results['book_value']:.2f}\n\n"
        
        # Valuation assessment
        if results['intrinsic_value'] > results['book_value'] * 1.2:
            summary += "**Valuation:** Significantly Undervalued ⭐⭐⭐\n\n"
        elif results['intrinsic_value'] > results['book_value'] * 1.05:
            summary += "**Valuation:** Slightly Undervalued ⭐⭐\n\n"
        elif results['intrinsic_value'] < results['book_value'] * 0.8:
            summary += "**Valuation:** Significantly Overvalued ⚠️⚠️⚠️\n\n"
        elif results['intrinsic_value'] < results['book_value'] * 0.95:
            summary += "**Valuation:** Slightly Overvalued ⚠️\n\n"
        else:
            summary += "**Valuation:** Fairly Valued ⭐\n\n"
        
        summary += "### Historical Data and Predictions:\n\n"
        summary += "| Date | Historical Price | LSTM | LLM | Hybrid |\n"
        summary += "|------|-----------------|------|-----|--------|\n"
        
        for i, date in enumerate(results['dates']):
            historical = results['historical_prices'][i] if i < len(results['historical_prices']) else "N/A"
            historical_str = f"${historical:.2f}" if isinstance(historical, (int, float)) else historical
            
            summary += f"| {date} | {historical_str} | ${results['lstm_predictions'][i]:.2f} | ${results['llm_predictions'][i]:.2f} | ${results['hybrid_predictions'][i]:.2f} |\n"
        
        if results['intersection_points']:
            summary += "\n### Intersection Points (where models agree):\n\n"
            for day, price in results['intersection_points']:
                date_idx = int(day)
                date = results['dates'][date_idx] if date_idx < len(results['dates']) else "Future"
                summary += f"- Day {day+1:.1f} ({date}): ${price:.2f}\n"
        
        # Create prediction data as JSON
        prediction_data = json.dumps(results)
        
        return fig, summary, prediction_data
        
    except Exception as e:
        logger.error(f"Error in predict_stock for {ticker}: {str(e)}")
        return None, f"Error: {str(e)}", None

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="Stock Price Prediction with Hybrid ML/LLM Approach") as demo:
        gr.Markdown("# Stock Price Prediction with Hybrid ML/LLM Approach")
        gr.Markdown("This app combines LSTM neural networks with LLM predictions to forecast stock prices.")
        
        with gr.Row():
            with gr.Column(scale=1):
                ticker_input = gr.Textbox(label="Stock Ticker Symbol", placeholder="e.g., AAPL, MSFT, GOOGL", value="AAPL")
                days_input = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Days to Predict")
                
                with gr.Row():
                    lstm_weight = gr.Slider(minimum=0, maximum=1, value=0.6, step=0.1, label="LSTM Weight")
                    llm_weight = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.1, label="LLM Weight")
                
                predict_btn = gr.Button("Predict", variant="primary")
                
            with gr.Column(scale=2):
                with gr.Tab("Visualization"):
                    plot_output = gr.Plot(label="Prediction Plot")
                
                with gr.Tab("Summary"):
                    summary_output = gr.Markdown(label="Prediction Summary")
                
                with gr.Tab("Raw Data"):
                    json_output = gr.JSON(label="Prediction Data")
        
        predict_btn.click(
            fn=predict_stock,
            inputs=[ticker_input, days_input, lstm_weight, llm_weight],
            outputs=[plot_output, summary_output, json_output]
        )
        
        gr.Markdown("""
        ## How it works
        
        1. **Stock Value Formula**: Calculates intrinsic value using Graham's formula: √(22.5 * EPS * Book Value)
        2. **LSTM Model**: Predicts based on historical price patterns
        3. **LLM Model**: Provides predictions based on fundamental analysis
        4. **Hybrid Approach**: Combines both predictions with configurable weights
        5. **Intersection Points**: Identifies where both models agree, potentially indicating strong signals
        
        ## Research Applications
        
        This hybrid approach demonstrates how traditional ML models can be combined with LLMs for financial forecasting.
        """)
        
    return demo

# Create and launch the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch()