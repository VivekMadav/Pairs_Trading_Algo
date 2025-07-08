"""
Configuration file for Pairs Trading Model
Contains all parameters and settings for easy modification
"""

import os
from datetime import datetime, timedelta

class Config:
    """Main configuration class for the pairs trading model"""
    
    # Data Collection Parameters
    DATA_FREQUENCY = '1d'  # '1d', '1h', '5m', etc.
    START_DATE = '2024-01-01'
    END_DATE = '2025-01-31'
    MIN_DATA_YEARS = 3  # Minimum years of data required
    
    # S&P 500 Universe Parameters
    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    MIN_MARKET_CAP = 1e10  # $10B minimum market cap
    MIN_AVG_VOLUME = 1e6   # 1M minimum average daily volume
    
    # Data Quality Parameters
    MAX_MISSING_DAYS = 30  # Maximum missing days allowed
    MIN_DATA_QUALITY_SCORE = 0.8  # Minimum data quality score (0-1)
    
    # File Paths
    DATA_DIR = 'data'
    LOGS_DIR = 'logs'
    RESULTS_DIR = 'results'
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Backtesting Parameters
    BACKTEST_START_DATE = '2022-01-01'
    BACKTEST_END_DATE = '2023-12-31'
    
    # Fail-safe Parameters
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30
    CHUNK_SIZE = 50  # Number of stocks to process in chunks
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [cls.DATA_DIR, cls.LOGS_DIR, cls.RESULTS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_data_path(cls, filename):
        """Get full path for data files"""
        return os.path.join(cls.DATA_DIR, filename)
    
    @classmethod
    def get_log_path(cls, filename):
        """Get full path for log files"""
        return os.path.join(cls.LOGS_DIR, filename)
    
    @classmethod
    def get_results_path(cls, filename):
        """Get full path for results files"""
        return os.path.join(cls.RESULTS_DIR, filename) 