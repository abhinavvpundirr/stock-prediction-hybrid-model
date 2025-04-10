Hybrid Stock Price Prediction

A machine learning pipeline that combines LSTM neural networks with Large Language Models (LLMs) for stock price prediction.

## Overview

This project implements a hybrid approach to stock price prediction by combining:
- Traditional LSTM models for technical analysis and pattern recognition
- LLM-based predictions (using Google's Flan-T5) for fundamental analysis
- Graham's intrinsic value formula for valuation assessment

The system identifies potential high-confidence signals at points where both models agree on price direction.

## Features

- **Hybrid Prediction Algorithm**: Weighted ensemble combining LSTM and LLM predictions
- **Intrinsic Value Calculation**: Implementation of Graham's formula `√(22.5 * EPS * Book Value)`
- **Intersection Analysis**: Identification of points where both models agree
- **Interactive UI**: Gradio-based interface for visualization and analysis
- **MLOps Pipeline**: Includes Docker containerization and DVC for versioning

## Technical Stack

- **Python 3.9**
- **TensorFlow/Keras** for LSTM implementation
- **Hugging Face Transformers** (Flan-T5) for LLM-based predictions
- **yfinance** for market data acquisition
- **Gradio** for interactive UI
- **Docker** for containerization
- **DVC** for data versioning
- **Pandas, NumPy, Matplotlib** for data processing and visualization

## Disclaimer

This project is for educational and research purposes only. The predictions should not be used for actual trading decisions. This is not financial advice, and no responsibility is taken for any investment decisions made based on this tool.

## Installation

# Clone the repository
git clone https://github.com/abhinavvpundirr/stock-prediction-hybrid-model
cd stock-prediction-hybrid-model

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
