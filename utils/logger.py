"""
Logging utilities.
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "Web-Content-Q-A-Tool-", level: int = logging.INFO):
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    return logger