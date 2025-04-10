import math
import logging

logger = logging.getLogger(__name__)

class StockValueCalculator:
    """
    Calculates intrinsic stock value using the formula: √(22.5 * EPS * Book Value)
    """
    def __init__(self):
        logger.info("Initializing StockValueCalculator")
        
    def calculate_intrinsic_value(self, eps: float, book_value: float) -> float:
        """
        Calculate the intrinsic value of a stock using Graham's formula
        
        Args:
            eps: Earnings Per Share
            book_value: Book Value per share
            
        Returns:
            float: Intrinsic value of the stock
        """
        if eps <= 0 or book_value <= 0:
            logger.warning(f"Invalid inputs: EPS={eps}, Book Value={book_value}")
            return 0
            
        intrinsic_value = math.sqrt(22.5 * eps * book_value)
        logger.info(f"Calculated intrinsic value: {intrinsic_value}")
        return intrinsic_value

