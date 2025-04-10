---
title: Stock Price Prediction with Hybrid ML/LLM Approach
emoji: 📈
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: 3.9
pinned: false
---

# Hybrid Stock Price Prediction

A machine learning pipeline that combines LSTM neural networks with Large Language Models (LLMs) for stock price prediction.

## Features

- **Hybrid Prediction Approach**: Combines traditional LSTM models with LLM-based predictions
- **Stock Value Formula**: Implements Graham's formula `√(22.5 * EPS * Book Value)` for intrinsic value calculation
- **Intersection Analysis**: Identifies points where both models agree, potentially indicating strong signals
- **Interactive UI**: Gradio-based interface for easy interaction and visualization
- **MLOps Best Practices**: Includes logging, testing, CI/CD, and experiment tracking
