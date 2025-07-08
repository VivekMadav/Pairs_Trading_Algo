"""
Logging utility for Pairs Trading Model
Provides centralized logging functionality
"""

import logging
import os
from datetime import datetime
from config import Config

class Logger:
    """Centralized logging class for the pairs trading model"""
    
    def __init__(self, name, log_file=None):
        """
        Initialize logger
        
        Args:
            name (str): Logger name
            log_file (str): Optional log file name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Create formatter
        formatter = logging.Formatter(Config.LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Config.get_log_path(log_file)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)

def get_logger(name, log_file=None):
    """
    Get logger instance
    
    Args:
        name (str): Logger name
        log_file (str): Optional log file name
    
    Returns:
        Logger: Logger instance
    """
    return Logger(name, log_file) 