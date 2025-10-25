from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(num1, num2):
    try:
        logger.info(f"Attempting to divide {num1} by {num2}")
        result = num1 / num2
        logger.info(f"Division successful: {result}")
        return result
    except Exception as e:
        logger.error("An error occurred during division")
        raise CustomException(str(e), sys) from e
    
if __name__ == "__main__":
    try:
        divide_numbers(10, 0)
    except CustomException as ce:
        logger.error(f"CustomException caught: {ce}")