import logging
from transformers import pipeline
import numpy as np
import os

logger = logging.getLogger(__name__)

class LLMStockPredictor:
    """
    LLM-based stock prediction using Hugging Face transformers
    """
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the LLM predictor
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        logger.info(f"Initializing LLM Stock Predictor with model {model_name}")
        self.model_name = model_name
        # Set environment variable to use CPU only to avoid CUDA issues
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        try:
            # Try to load a smaller model by default
            self.pipeline = pipeline("text2text-generation", model=model_name)
            logger.info("LLM pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM pipeline with {model_name}: {str(e)}")
            # Fallback to an even smaller model if available
            try:
                logger.info("Attempting to initialize with distilgpt2 as fallback")
                self.pipeline = pipeline("text-generation", model="distilgpt2")
                logger.info("Fallback LLM pipeline initialized successfully")
            except Exception as e2:
                logger.error(f"Error initializing fallback LLM pipeline: {str(e2)}")
                self.pipeline = None
    
    def generate_prompt(self, ticker: str, current_price: float, eps: float, book_value: float, 
                        intrinsic_value: float, days: int = 10) -> str:
        """
        Generate a prompt for the LLM
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            eps: Earnings Per Share
            book_value: Book Value per share
            intrinsic_value: Calculated intrinsic value
            days: Number of days to predict
            
        Returns:
            str: Formatted prompt for the LLM
        """
        prompt = f"""
        As a financial expert, analyze the stock {ticker} with the following information:
        - Current price: ${current_price:.2f}
        - Earnings Per Share (EPS): ${eps:.2f}
        - Book Value per share: ${book_value:.2f}
        - Calculated intrinsic value using Graham's formula (√(22.5 * EPS * Book Value)): ${intrinsic_value:.2f}
        
        Based on this information, predict the stock price for the next {days} trading days.
        Format your response as a comma-separated list of predicted prices only, without any additional text.
        """
        return prompt
    
    def predict(self, ticker: str, current_price: float, eps: float, book_value: float, 
                days_to_predict: int = 10) -> list:
        """
        Make predictions using the LLM
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            eps: Earnings Per Share
            book_value: Book Value per share
            days_to_predict: Number of days to predict
            
        Returns:
            list: Predicted stock prices
        """
        if self.pipeline is None:
            logger.error("LLM pipeline not initialized")
            # Generate fallback predictions instead of returning an empty list
            return self._generate_fallback_predictions(current_price, eps, book_value, days_to_predict)
        
        logger.info(f"Generating LLM prediction for {ticker} for next {days_to_predict} days")
        
        # Calculate intrinsic value
        try:
            from src.stock_formula import StockValueCalculator
            calculator = StockValueCalculator()
            intrinsic_value = calculator.calculate_intrinsic_value(eps, book_value)
        except ImportError:
            # Handle case where the import path might be different
            intrinsic_value = (22.5 * eps * book_value) ** 0.5
            logger.warning("Using direct calculation for intrinsic value due to import error")
        
        # Generate prompt and get prediction
        prompt = self.generate_prompt(ticker, current_price, eps, book_value, intrinsic_value, days_to_predict)
        
        try:
            response = self.pipeline(prompt, max_length=100)[0]['generated_text']
            logger.info(f"LLM response: {response}")
            
            # Parse the response to extract predicted prices
            predictions = []
            try:
                # Try to parse as comma-separated values
                values = response.strip().split(',')
                for val in values:
                    # Clean and convert to float
                    clean_val = val.strip().replace('$', '').replace(' ', '')
                    if clean_val:
                        try:
                            predictions.append(float(clean_val))
                        except ValueError:
                            # Skip values that can't be converted to float
                            continue
                
                # If we didn't get enough predictions, generate fallback predictions
                if len(predictions) < days_to_predict:
                    fallback = self._generate_fallback_predictions(
                        current_price, eps, book_value, days_to_predict - len(predictions)
                    )
                    predictions.extend(fallback)
                
                # If we got too many predictions, truncate
                predictions = predictions[:days_to_predict]
                
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                # Fallback: generate simple predictions 
                predictions = self._generate_fallback_predictions(current_price, eps, book_value, days_to_predict)
            
            logger.info(f"Parsed predictions: {predictions}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting LLM prediction: {str(e)}")
            return self._generate_fallback_predictions(current_price, eps, book_value, days_to_predict)
    
    def _generate_fallback_predictions(self, current_price: float, eps: float, book_value: float, days: int) -> list:
        """
        Generate fallback predictions when LLM fails
        
        Args:
            current_price: Current stock price
            eps: Earnings Per Share
            book_value: Book Value per share
            days: Number of days to predict
            
        Returns:
            list: Predicted stock prices
        """
        logger.info("Generating fallback predictions")
        
        # Calculate intrinsic value
        try:
            intrinsic_value = (22.5 * eps * book_value) ** 0.5
        except:
            intrinsic_value = current_price
        
        # Add some randomness to make predictions more realistic
        np.random.seed(42)  # For reproducibility
        
        if intrinsic_value > current_price:
            # Stock is undervalued, predict gradual increase
            step = (intrinsic_value - current_price) / (days * 2)
            base_predictions = [current_price + step * i for i in range(1, days + 1)]
        else:
            # Stock is overvalued, predict gradual decrease
            step = (current_price - intrinsic_value) / (days * 2)
            base_predictions = [current_price - step * i for i in range(1, days + 1)]
        
        # Add some random noise (up to 2%)
        noise = np.random.uniform(-0.02, 0.02, days) * current_price
        predictions = [max(0.01, base + n) for base, n in zip(base_predictions, noise)]
        
        logger.info(f"Fallback predictions: {predictions}")
        return predictions