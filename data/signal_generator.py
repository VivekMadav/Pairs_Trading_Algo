"""
Signal Generation for Pairs Trading Model
Handles z-score based signal generation, position sizing, and trade management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import Config

class SignalGenerator:
    """Generates trading signals for pairs trading based on z-score thresholds"""
    
    def __init__(self):
        """Initialize SignalGenerator"""
        self.logger = get_logger('SignalGenerator', 'signal_generator.log')
        self.price_data = None
        self.pairs_data = None
        self.signals = None
    
    def load_price_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load price data from Phase 1
        
        Args:
            data_path (str): Path to combined price data file
            
        Returns:
            pd.DataFrame: Price data with ticker and date columns
        """
        try:
            if data_path is None:
                data_path = Config.get_data_path('phase1_price_data_combined.csv')
            
            self.logger.info(f"Loading price data from {data_path}")
            
            # Load combined price data
            price_data = pd.read_csv(data_path)
            
            # Convert date column to datetime
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            # Ensure we have the required columns
            required_cols = ['date', 'ticker', 'close']
            if not all(col in price_data.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Expected: {required_cols}")
            
            self.logger.info(f"Loaded price data for {price_data['ticker'].nunique()} tickers")
            self.logger.info(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")
            
            self.price_data = price_data
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error loading price data: {str(e)}")
            raise
    
    def load_pairs_data(self, pairs_path: str = None) -> pd.DataFrame:
        """
        Load pairs data from Phase 3
        
        Args:
            pairs_path (str): Path to pairs data file
            
        Returns:
            pd.DataFrame: Pairs data
        """
        try:
            if pairs_path is None:
                pairs_path = Config.get_data_path('phase2_selected_pairs.csv')
            
            self.logger.info(f"Loading pairs data from {pairs_path}")
            
            pairs_data = pd.read_csv(pairs_path)
            
            self.logger.info(f"Loaded {len(pairs_data)} pairs for signal generation")
            
            self.pairs_data = pairs_data
            return pairs_data
            
        except Exception as e:
            self.logger.error(f"Error loading pairs data: {str(e)}")
            raise
    
    def calculate_spread(self, 
                        ticker1: str,
                        ticker2: str,
                        hedge_ratio: float = None) -> pd.Series:
        """
        Calculate spread between two stocks
        
        Args:
            ticker1 (str): First ticker
            ticker2 (str): Second ticker
            hedge_ratio (float): Hedge ratio (if None, will calculate)
            
        Returns:
            pd.Series: Spread series
        """
        try:
            # Get price data for both tickers
            ticker1_data = self.price_data[self.price_data['ticker'] == ticker1].set_index('date')['close']
            ticker2_data = self.price_data[self.price_data['ticker'] == ticker2].set_index('date')['close']
            
            # Align data
            aligned_data = pd.concat([ticker1_data, ticker2_data], axis=1).dropna()
            aligned_data.columns = [ticker1, ticker2]
            
            if len(aligned_data) < 50:
                raise ValueError(f"Insufficient data for {ticker1}-{ticker2}: {len(aligned_data)} observations")
            
            # Calculate hedge ratio if not provided
            if hedge_ratio is None:
                # Simple OLS regression
                X = aligned_data[ticker2].values.reshape(-1, 1)
                y = aligned_data[ticker1].values
                hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]
            
            # Calculate spread
            spread = aligned_data[ticker1] - hedge_ratio * aligned_data[ticker2]
            
            self.logger.info(f"Calculated spread for {ticker1}-{ticker2}: hedge_ratio={hedge_ratio:.4f}")
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Error calculating spread for {ticker1}-{ticker2}: {str(e)}")
            raise
    
    def calculate_z_score(self, 
                         spread: pd.Series,
                         window: int = 20) -> pd.Series:
        """
        Calculate rolling z-score of spread
        
        Args:
            spread (pd.Series): Spread series
            window (int): Rolling window size
            
        Returns:
            pd.Series: Z-score series
        """
        try:
            # Calculate rolling mean and standard deviation
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()
            
            # Calculate z-score
            z_score = (spread - rolling_mean) / rolling_std
            
            self.logger.info(f"Calculated z-score with {window}-day rolling window")
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating z-score: {str(e)}")
            raise
    
    def generate_signals(self, 
                        z_score: pd.Series,
                        entry_threshold: float = 2.0,
                        exit_threshold: float = 0.5,
                        stop_loss_threshold: float = 4.0) -> pd.Series:
        """
        Generate trading signals based on z-score
        
        Args:
            z_score (pd.Series): Z-score series
            entry_threshold (float): Entry threshold (absolute value)
            exit_threshold (float): Exit threshold (absolute value)
            stop_loss_threshold (float): Stop loss threshold (absolute value)
            
        Returns:
            pd.Series: Signal series (1: long spread, -1: short spread, 0: no position)
        """
        try:
            signals = pd.Series(0, index=z_score.index)
            
            position = 0  # Current position: 0 (flat), 1 (long spread), -1 (short spread)
            
            for i, z in enumerate(z_score):
                if pd.isna(z):
                    continue
                
                # Entry signals
                if position == 0:  # No position
                    if z > entry_threshold:
                        position = -1  # Short spread (sell ticker1, buy ticker2)
                        signals.iloc[i] = -1
                    elif z < -entry_threshold:
                        position = 1   # Long spread (buy ticker1, sell ticker2)
                        signals.iloc[i] = 1
                
                # Exit signals
                elif position == 1:  # Long spread position
                    if abs(z) < exit_threshold or z > stop_loss_threshold:
                        position = 0
                        signals.iloc[i] = 0
                    else:
                        signals.iloc[i] = 1  # Maintain position
                
                elif position == -1:  # Short spread position
                    if abs(z) < exit_threshold or z < -stop_loss_threshold:
                        position = 0
                        signals.iloc[i] = 0
                    else:
                        signals.iloc[i] = -1  # Maintain position
            
            self.logger.info(f"Generated signals: {len(signals[signals != 0])} trades")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def calculate_positions(self, 
                           signals: pd.Series,
                           ticker1: str,
                           ticker2: str,
                           hedge_ratio: float) -> Dict[str, pd.Series]:
        """
        Calculate actual positions for each stock
        
        Args:
            signals (pd.Series): Signal series
            ticker1 (str): First ticker
            ticker2 (str): Second ticker
            hedge_ratio (float): Hedge ratio
            
        Returns:
            Dict[str, pd.Series]: Position series for each ticker
        """
        try:
            # Initialize position series
            pos1 = pd.Series(0, index=signals.index)
            pos2 = pd.Series(0, index=signals.index)
            
            # Calculate positions based on signals
            for i, signal in enumerate(signals):
                if signal == 1:  # Long spread
                    pos1.iloc[i] = 1
                    pos2.iloc[i] = -hedge_ratio
                elif signal == -1:  # Short spread
                    pos1.iloc[i] = -1
                    pos2.iloc[i] = hedge_ratio
                else:  # No position
                    pos1.iloc[i] = 0
                    pos2.iloc[i] = 0
            
            positions = {
                ticker1: pos1,
                ticker2: pos2
            }
            
            self.logger.info(f"Calculated positions for {ticker1} and {ticker2}")
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error calculating positions: {str(e)}")
            raise
    
    def calculate_returns(self, 
                         positions: Dict[str, pd.Series],
                         ticker1: str,
                         ticker2: str) -> pd.Series:
        """
        Calculate strategy returns
        
        Args:
            positions (Dict[str, pd.Series]): Position series for each ticker
            ticker1 (str): First ticker
            ticker2 (str): Second ticker
            
        Returns:
            pd.Series: Strategy returns
        """
        try:
            # Get price data
            ticker1_data = self.price_data[self.price_data['ticker'] == ticker1].set_index('date')['close']
            ticker2_data = self.price_data[self.price_data['ticker'] == ticker2].set_index('date')['close']
            
            # Align data
            aligned_data = pd.concat([ticker1_data, ticker2_data], axis=1).dropna()
            aligned_data.columns = [ticker1, ticker2]
            
            # Calculate returns
            returns1 = aligned_data[ticker1].pct_change()
            returns2 = aligned_data[ticker2].pct_change()
            
            # Calculate strategy returns
            strategy_returns = (positions[ticker1] * returns1 + 
                              positions[ticker2] * returns2)
            
            self.logger.info(f"Calculated strategy returns for {ticker1}-{ticker2}")
            
            return strategy_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def generate_all_signals(self, 
                           pairs_data: pd.DataFrame,
                           entry_threshold: float = 2.0,
                           exit_threshold: float = 0.5,
                           stop_loss_threshold: float = 4.0) -> List[Dict]:
        """
        Generate signals for all pairs
        
        Args:
            pairs_data (pd.DataFrame): Pairs data
            entry_threshold (float): Entry threshold
            exit_threshold (float): Exit threshold
            stop_loss_threshold (float): Stop loss threshold
            
        Returns:
            List[Dict]: Signal results for all pairs
        """
        try:
            self.logger.info("Generating signals for all pairs...")
            
            results = []
            
            for _, pair in pairs_data.iterrows():
                ticker1, ticker2 = pair['ticker1'], pair['ticker2']
                
                try:
                    self.logger.info(f"Generating signals for {ticker1}-{ticker2}")
                    
                    # Calculate spread
                    spread = self.calculate_spread(ticker1, ticker2)
                    
                    # Calculate z-score
                    z_score = self.calculate_z_score(spread)
                    
                    # Generate signals
                    signals = self.generate_signals(
                        z_score, 
                        entry_threshold, 
                        exit_threshold, 
                        stop_loss_threshold
                    )
                    
                    # Calculate hedge ratio
                    hedge_ratio = self._calculate_hedge_ratio(ticker1, ticker2)
                    
                    # Calculate positions
                    positions = self.calculate_positions(signals, ticker1, ticker2, hedge_ratio)
                    
                    # Calculate returns
                    strategy_returns = self.calculate_returns(positions, ticker1, ticker2)
                    
                    # Store results
                    result = {
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'correlation': pair['correlation'],
                        'spread': spread,
                        'z_score': z_score,
                        'signals': signals,
                        'positions': positions,
                        'strategy_returns': strategy_returns,
                        'hedge_ratio': hedge_ratio,
                        'entry_threshold': entry_threshold,
                        'exit_threshold': exit_threshold,
                        'stop_loss_threshold': stop_loss_threshold
                    }
                    
                    results.append(result)
                    
                    self.logger.info(f"Completed signal generation for {ticker1}-{ticker2}")
                    
                except Exception as e:
                    self.logger.error(f"Error generating signals for {ticker1}-{ticker2}: {str(e)}")
                    continue
            
            self.signals = results
            self.logger.info(f"Generated signals for {len(results)} pairs")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating all signals: {str(e)}")
            raise
    
    def _calculate_hedge_ratio(self, ticker1: str, ticker2: str) -> float:
        """
        Calculate hedge ratio using OLS regression
        
        Args:
            ticker1 (str): First ticker
            ticker2 (str): Second ticker
            
        Returns:
            float: Hedge ratio
        """
        try:
            # Get price data
            ticker1_data = self.price_data[self.price_data['ticker'] == ticker1].set_index('date')['close']
            ticker2_data = self.price_data[self.price_data['ticker'] == ticker2].set_index('date')['close']
            
            # Align data
            aligned_data = pd.concat([ticker1_data, ticker2_data], axis=1).dropna()
            aligned_data.columns = [ticker1, ticker2]
            
            # Calculate hedge ratio using OLS
            X = aligned_data[ticker2].values.reshape(-1, 1)
            y = aligned_data[ticker1].values
            hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge ratio: {str(e)}")
            raise
    
    def save_results(self, 
                    results: List[Dict],
                    filename: str = 'signal_generation_results.json') -> None:
        """
        Save signal generation results
        
        Args:
            results (List[Dict]): Signal results
            filename (str): Output filename
        """
        try:
            import json
            
            output_path = Config.get_results_path(filename)
            
            # Create simplified results structure
            simplified_results = []
            for result in results:
                simplified_result = {
                    'ticker1': result['ticker1'],
                    'ticker2': result['ticker2'],
                    'correlation': float(result['correlation']),
                    'hedge_ratio': float(result['hedge_ratio']),
                    'entry_threshold': float(result['entry_threshold']),
                    'exit_threshold': float(result['exit_threshold']),
                    'stop_loss_threshold': float(result['stop_loss_threshold']),
                    'total_trades': int((result['signals'] != 0).sum()),
                    'long_trades': int((result['signals'] == 1).sum()),
                    'short_trades': int((result['signals'] == -1).sum()),
                    'avg_z_score': float(result['z_score'].mean()),
                    'z_score_std': float(result['z_score'].std())
                }
                simplified_results.append(simplified_result)
            
            with open(output_path, 'w') as f:
                json.dump(simplified_results, f, indent=2)
            
            self.logger.info(f"Saved signal generation results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise 