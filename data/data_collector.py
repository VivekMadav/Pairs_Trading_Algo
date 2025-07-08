"""
Data Collector for Pairs Trading Model
Handles fetching and cleaning historical price data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import Config

class DataCollector:
    """Handles collection and cleaning of historical price data"""
    
    def __init__(self):
        """Initialize DataCollector"""
        self.logger = get_logger('DataCollector', 'data_collector.log')
        self.price_data = {}
        self.data_quality_scores = {}
    
    def fetch_historical_data(self, 
                            tickers: List[str],
                            start_date: str = None,
                            end_date: str = None,
                            frequency: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for given tickers
        
        Args:
            tickers (List[str]): List of ticker symbols
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
            frequency (str): Data frequency ('1d', '1h', etc.)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of price data by ticker
        """
        try:
            # Use config defaults if not provided
            start_date = start_date or Config.START_DATE
            end_date = end_date or Config.END_DATE
            frequency = frequency or Config.DATA_FREQUENCY
            
            self.logger.info(f"Fetching historical data for {len(tickers)} tickers")
            self.logger.info(f"Period: {start_date} to {end_date}, Frequency: {frequency}")
            
            price_data = {}
            
            for i, ticker in enumerate(tickers):
                try:
                    self.logger.info(f"Fetching data for {ticker} ({i+1}/{len(tickers)})")
                    
                    # Fetch data using yfinance
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date, interval=frequency)
                    
                    if data.empty:
                        self.logger.warning(f"No data found for {ticker}")
                        continue
                    
                    # Clean column names
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Add ticker column
                    data['ticker'] = ticker
                    
                    # Reset index to make date a column
                    data = data.reset_index()
                    # Ensure the date column is named 'date'
                    if 'Date' in data.columns:
                        data = data.rename(columns={'Date': 'date'})
                    elif 'date' not in data.columns:
                        # If neither, raise an error
                        raise KeyError("No date column found after resetting index.")
                    
                    price_data[ticker] = data
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                    continue
            
            self.price_data = price_data
            self.logger.info(f"Successfully fetched data for {len(price_data)} tickers")
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error in fetch_historical_data: {str(e)}")
            raise
    
    def clean_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean and validate price data
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Raw price data
            
        Returns:
            Dict[str, pd.DataFrame]: Cleaned price data
        """
        try:
            self.logger.info("Cleaning price data...")
            
            cleaned_data = {}
            
            for ticker, data in price_data.items():
                try:
                    # Remove rows with missing values
                    data_clean = data.dropna()
                    
                    # Remove rows with zero or negative prices
                    data_clean = data_clean[
                        (data_clean['open'] > 0) &
                        (data_clean['high'] > 0) &
                        (data_clean['low'] > 0) &
                        (data_clean['close'] > 0)
                    ]
                    
                    # Remove rows with zero volume
                    data_clean = data_clean[data_clean['volume'] > 0]
                    
                    # Sort by date
                    data_clean = data_clean.sort_values('date')
                    
                    # Reset index
                    data_clean = data_clean.reset_index(drop=True)
                    
                    cleaned_data[ticker] = data_clean
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning data for {ticker}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully cleaned data for {len(cleaned_data)} tickers")
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error in clean_data: {str(e)}")
            raise
    
    def calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate data quality score for a ticker
        
        Args:
            data (pd.DataFrame): Price data for a ticker
            
        Returns:
            float: Data quality score (0-1)
        """
        try:
            if data.empty:
                return 0.0
            
            # Calculate expected number of trading days
            start_date = data['date'].min()
            end_date = data['date'].max()
            expected_days = len(pd.bdate_range(start=start_date, end=end_date))
            
            # Calculate actual number of days
            actual_days = len(data)
            
            # Calculate completeness score
            completeness_score = actual_days / expected_days if expected_days > 0 else 0
            
            # Calculate consistency score (no large gaps)
            date_diffs = data['date'].diff().dt.days
            max_gap = date_diffs.max() if len(date_diffs) > 1 else 0
            consistency_score = 1.0 if max_gap <= Config.MAX_MISSING_DAYS else 0.5
            
            # Calculate overall quality score
            quality_score = (completeness_score + consistency_score) / 2
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {str(e)}")
            return 0.0
    
    def filter_by_quality(self, 
                         price_data: Dict[str, pd.DataFrame],
                         min_quality_score: float = None) -> Dict[str, pd.DataFrame]:
        """
        Filter data based on quality score
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data
            min_quality_score (float): Minimum quality score threshold
            
        Returns:
            Dict[str, pd.DataFrame]: Filtered price data
        """
        try:
            min_quality_score = min_quality_score or Config.MIN_DATA_QUALITY_SCORE
            
            self.logger.info(f"Filtering data by quality score (min: {min_quality_score})")
            
            filtered_data = {}
            quality_scores = {}
            
            for ticker, data in price_data.items():
                quality_score = self.calculate_data_quality_score(data)
                quality_scores[ticker] = quality_score
                
                if quality_score >= min_quality_score:
                    filtered_data[ticker] = data
                else:
                    self.logger.warning(f"Excluding {ticker} due to low quality score: {quality_score:.3f}")
            
            self.data_quality_scores = quality_scores
            self.logger.info(f"Quality filtering complete: {len(filtered_data)} tickers passed")
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error in filter_by_quality: {str(e)}")
            raise
    
    def save_data(self, 
                  price_data: Dict[str, pd.DataFrame],
                  filename_prefix: str = 'price_data') -> None:
        """
        Save price data to files
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data to save
            filename_prefix (str): Prefix for output files
        """
        try:
            self.logger.info("Saving price data to files...")
            
            # Save individual ticker files
            for ticker, data in price_data.items():
                filename = f"{filename_prefix}_{ticker}.csv"
                filepath = Config.get_data_path(filename)
                data.to_csv(filepath, index=False)
            
            # Save combined data
            combined_data = pd.concat(price_data.values(), ignore_index=True)
            combined_filename = f"{filename_prefix}_combined.csv"
            combined_filepath = Config.get_data_path(combined_filename)
            combined_data.to_csv(combined_filepath, index=False)
            
            # Save quality scores
            if self.data_quality_scores:
                quality_df = pd.DataFrame([
                    {'ticker': ticker, 'quality_score': score}
                    for ticker, score in self.data_quality_scores.items()
                ])
                quality_filename = f"{filename_prefix}_quality_scores.csv"
                quality_filepath = Config.get_data_path(quality_filename)
                quality_df.to_csv(quality_filepath, index=False)
            
            self.logger.info(f"Successfully saved data to {Config.DATA_DIR}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def get_data_summary(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for price data
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        try:
            summary_data = []
            
            for ticker, data in price_data.items():
                if data.empty:
                    continue
                
                summary = {
                    'ticker': ticker,
                    'start_date': data['date'].min(),
                    'end_date': data['date'].max(),
                    'total_days': len(data),
                    'avg_price': data['close'].mean(),
                    'avg_volume': data['volume'].mean(),
                    'price_volatility': data['close'].std(),
                    'quality_score': self.data_quality_scores.get(ticker, 0.0)
                }
                
                summary_data.append(summary)
            
            summary_df = pd.DataFrame(summary_data)
            return summary_df
            
        except Exception as e:
            self.logger.error(f"Error generating data summary: {str(e)}")
            raise 