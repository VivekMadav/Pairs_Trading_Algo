"""
S&P 500 Universe Management
Handles fetching and filtering S&P 500 constituents
"""

import pandas as pd
import requests
import time
from typing import List, Dict, Optional
from utils.logger import get_logger
from config import Config

class SP500Universe:
    """Manages S&P 500 stock universe and metadata"""
    
    def __init__(self):
        """Initialize SP500Universe"""
        self.logger = get_logger('SP500Universe', 'sp500_universe.log')
        self.constituents = None
        self.metadata = None
    
    def fetch_sp500_constituents(self) -> pd.DataFrame:
        """
        Fetch current S&P 500 constituents from Wikipedia
        
        Returns:
            pd.DataFrame: DataFrame with ticker symbols and company info
        """
        try:
            self.logger.info("Fetching S&P 500 constituents from Wikipedia...")
            
            # Fetch S&P 500 table from Wikipedia
            tables = pd.read_html(Config.SP500_URL)
            sp500_table = tables[0]  # First table contains the constituents
            
            # Clean column names
            sp500_table.columns = sp500_table.columns.str.strip()
            
            # Extract relevant columns
            constituents = sp500_table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].copy()
            constituents.columns = ['ticker', 'company_name', 'sector', 'sub_industry']
            
            # Clean ticker symbols
            constituents['ticker'] = constituents['ticker'].str.strip()
            
            self.logger.info(f"Successfully fetched {len(constituents)} S&P 500 constituents")
            self.constituents = constituents
            
            return constituents
            
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 constituents: {str(e)}")
            raise
    
    def fetch_stock_metadata(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch metadata for given tickers using yfinance
        
        Args:
            tickers (List[str]): List of ticker symbols
            
        Returns:
            pd.DataFrame: DataFrame with stock metadata
        """
        import yfinance as yf
        
        try:
            self.logger.info(f"Fetching metadata for {len(tickers)} stocks...")
            
            metadata_list = []
            
            for i in range(0, len(tickers), Config.CHUNK_SIZE):
                chunk = tickers[i:i + Config.CHUNK_SIZE]
                self.logger.info(f"Processing chunk {i//Config.CHUNK_SIZE + 1}/{(len(tickers)-1)//Config.CHUNK_SIZE + 1}")
                
                for ticker in chunk:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        metadata = {
                            'ticker': ticker,
                            'market_cap': info.get('marketCap', 0),
                            'avg_volume': info.get('averageVolume', 0),
                            'sector': info.get('sector', 'Unknown'),
                            'industry': info.get('industry', 'Unknown'),
                            'country': info.get('country', 'Unknown'),
                            'currency': info.get('currency', 'USD')
                        }
                        
                        metadata_list.append(metadata)
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.warning(f"Error fetching metadata for {ticker}: {str(e)}")
                        continue
            
            metadata_df = pd.DataFrame(metadata_list)
            self.logger.info(f"Successfully fetched metadata for {len(metadata_df)} stocks")
            self.metadata = metadata_df
            
            return metadata_df
            
        except Exception as e:
            self.logger.error(f"Error fetching stock metadata: {str(e)}")
            raise
    
    def filter_stocks(self, 
                     min_market_cap: Optional[float] = None,
                     min_avg_volume: Optional[float] = None) -> pd.DataFrame:
        """
        Filter stocks based on criteria
        
        Args:
            min_market_cap (float): Minimum market cap filter
            min_avg_volume (float): Minimum average volume filter
            
        Returns:
            pd.DataFrame: Filtered stock universe
        """
        try:
            if self.metadata is None:
                raise ValueError("Metadata not available. Run fetch_stock_metadata first.")
            
            filtered_stocks = self.metadata.copy()
            
            # Apply market cap filter
            if min_market_cap is not None:
                filtered_stocks = filtered_stocks[filtered_stocks['market_cap'] >= min_market_cap]
                self.logger.info(f"Applied market cap filter: {len(filtered_stocks)} stocks remaining")
            
            # Apply volume filter
            if min_avg_volume is not None:
                filtered_stocks = filtered_stocks[filtered_stocks['avg_volume'] >= min_avg_volume]
                self.logger.info(f"Applied volume filter: {len(filtered_stocks)} stocks remaining")
            
            return filtered_stocks
            
        except Exception as e:
            self.logger.error(f"Error filtering stocks: {str(e)}")
            raise
    
    def get_filtered_universe(self) -> pd.DataFrame:
        """
        Get filtered S&P 500 universe based on config criteria
        
        Returns:
            pd.DataFrame: Filtered stock universe
        """
        try:
            # Fetch constituents if not already done
            if self.constituents is None:
                self.fetch_sp500_constituents()
            
            # Get tickers
            tickers = self.constituents['ticker'].tolist()
            
            # Fetch metadata
            metadata = self.fetch_stock_metadata(tickers)
            
            # Filter based on config criteria
            filtered_universe = self.filter_stocks(
                min_market_cap=Config.MIN_MARKET_CAP,
                min_avg_volume=Config.MIN_AVG_VOLUME
            )
            
            # Save to file
            output_path = Config.get_data_path('sp500_filtered_universe.csv')
            filtered_universe.to_csv(output_path, index=False)
            self.logger.info(f"Saved filtered universe to {output_path}")
            
            return filtered_universe
            
        except Exception as e:
            self.logger.error(f"Error getting filtered universe: {str(e)}")
            raise
    
    def save_universe(self, filename: str = 'sp500_universe.csv'):
        """
        Save current universe to file
        
        Args:
            filename (str): Output filename
        """
        try:
            if self.constituents is None:
                raise ValueError("No universe data to save")
            
            output_path = Config.get_data_path(filename)
            self.constituents.to_csv(output_path, index=False)
            self.logger.info(f"Saved universe to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving universe: {str(e)}")
            raise 