"""
Correlation Analysis for Pairs Trading Model
Handles correlation matrix computation and pair selection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import Config

class CorrelationAnalyzer:
    """Analyzes correlations between stocks and selects optimal pairs"""
    
    def __init__(self):
        """Initialize CorrelationAnalyzer"""
        self.logger = get_logger('CorrelationAnalyzer', 'correlation_analyzer.log')
        self.correlation_matrix = None
        self.returns_data = None
        self.selected_pairs = None
    
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
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error loading price data: {str(e)}")
            raise
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data
        
        Args:
            price_data (pd.DataFrame): Price data with date, ticker, close columns
            
        Returns:
            pd.DataFrame: Returns data with date and ticker columns
        """
        try:
            self.logger.info("Calculating daily returns...")
            
            # Sort by ticker and date
            price_data = price_data.sort_values(['ticker', 'date'])
            
            # Calculate log returns
            returns_data = price_data.groupby('ticker').apply(
                lambda x: x.assign(
                    returns=np.log(x['close'] / x['close'].shift(1))
                )
            ).reset_index(drop=True)
            
            # Remove first row for each ticker (NaN return)
            returns_data = returns_data.dropna(subset=['returns'])
            
            # Pivot to wide format (dates as index, tickers as columns)
            returns_wide = returns_data.pivot(
                index='date', 
                columns='ticker', 
                values='returns'
            )
            
            self.logger.info(f"Calculated returns for {returns_wide.shape[1]} tickers over {returns_wide.shape[0]} days")
            
            self.returns_data = returns_wide
            return returns_wide
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def compute_correlation_matrix(self, 
                                 returns_data: pd.DataFrame,
                                 method: str = 'pearson',
                                 min_periods: int = None) -> pd.DataFrame:
        """
        Compute correlation matrix for all stock pairs
        
        Args:
            returns_data (pd.DataFrame): Returns data (dates as index, tickers as columns)
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            min_periods (int): Minimum number of observations required
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        try:
            min_periods = min_periods or int(len(returns_data) * 0.8)  # 80% of data
            
            self.logger.info(f"Computing {method} correlation matrix...")
            self.logger.info(f"Minimum periods required: {min_periods}")
            
            # Compute correlation matrix
            correlation_matrix = returns_data.corr(
                method=method,
                min_periods=min_periods
            )
            
            # Remove self-correlations (set diagonal to NaN)
            np.fill_diagonal(correlation_matrix.values, np.nan)
            
            self.logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
            self.logger.info(f"Correlation range: {correlation_matrix.min().min():.3f} to {correlation_matrix.max().max():.3f}")
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing correlation matrix: {str(e)}")
            raise
    
    def get_all_pairs(self, correlation_matrix: pd.DataFrame) -> List[Dict]:
        """
        Get all possible pairs with their correlations
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            
        Returns:
            List[Dict]: List of pairs with correlation information
        """
        try:
            self.logger.info("Extracting all pairs from correlation matrix...")
            
            pairs = []
            
            # Get upper triangle of correlation matrix (avoid duplicates)
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Convert to pairs
            for ticker1 in upper_triangle.columns:
                for ticker2 in upper_triangle.index:
                    correlation = upper_triangle.loc[ticker2, ticker1]
                    
                    if pd.notna(correlation):
                        pair = {
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': correlation,
                            'abs_correlation': abs(correlation)
                        }
                        pairs.append(pair)
            
            # Sort by absolute correlation (descending)
            pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            self.logger.info(f"Extracted {len(pairs)} pairs")
            self.logger.info(f"Top correlation: {pairs[0]['correlation']:.3f} ({pairs[0]['ticker1']}-{pairs[0]['ticker2']})")
            self.logger.info(f"Bottom correlation: {pairs[-1]['correlation']:.3f} ({pairs[-1]['ticker1']}-{pairs[-1]['ticker2']})")
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error extracting pairs: {str(e)}")
            raise
    
    def filter_pairs(self, 
                    pairs: List[Dict],
                    min_correlation: float = 0.7,
                    max_correlation: float = 0.99,
                    exclude_same_sector: bool = True,
                    sector_data: pd.DataFrame = None) -> List[Dict]:
        """
        Filter pairs based on criteria
        
        Args:
            pairs (List[Dict]): List of pairs
            min_correlation (float): Minimum correlation threshold
            max_correlation (float): Maximum correlation threshold (avoid near-perfect correlation)
            exclude_same_sector (bool): Whether to exclude pairs from same sector
            sector_data (pd.DataFrame): Sector information for stocks
            
        Returns:
            List[Dict]: Filtered pairs
        """
        try:
            self.logger.info(f"Filtering pairs with correlation range: {min_correlation} to {max_correlation}")
            
            filtered_pairs = []
            
            for pair in pairs:
                correlation = pair['correlation']
                
                # Apply correlation filters
                if abs(correlation) < min_correlation or abs(correlation) > max_correlation:
                    continue
                
                # Apply sector filter if requested
                if exclude_same_sector and sector_data is not None:
                    ticker1_sector = sector_data.get(pair['ticker1'], {}).get('sector', 'Unknown')
                    ticker2_sector = sector_data.get(pair['ticker2'], {}).get('sector', 'Unknown')
                    
                    if ticker1_sector == ticker2_sector:
                        continue
                
                filtered_pairs.append(pair)
            
            self.logger.info(f"Filtered pairs: {len(filtered_pairs)} out of {len(pairs)}")
            
            return filtered_pairs
            
        except Exception as e:
            self.logger.error(f"Error filtering pairs: {str(e)}")
            raise
    
    def select_top_pairs(self, 
                        pairs: List[Dict],
                        n_pairs: int = 5,
                        diversity_threshold: float = 0.1) -> List[Dict]:
        """
        Select top pairs with diversity constraints
        
        Args:
            pairs (List[Dict]): List of filtered pairs
            n_pairs (int): Number of pairs to select
            diversity_threshold (float): Minimum correlation difference between selected pairs
            
        Returns:
            List[Dict]: Selected pairs
        """
        try:
            self.logger.info(f"Selecting top {n_pairs} pairs with diversity constraints...")
            
            selected_pairs = []
            used_tickers = set()
            
            for pair in pairs:
                # Check if we have enough pairs
                if len(selected_pairs) >= n_pairs:
                    break
                
                ticker1, ticker2 = pair['ticker1'], pair['ticker2']
                
                # Check if tickers are already used
                if ticker1 in used_tickers or ticker2 in used_tickers:
                    continue
                
                # Check diversity with already selected pairs
                diverse = True
                for selected_pair in selected_pairs:
                    correlation_diff = abs(pair['correlation'] - selected_pair['correlation'])
                    if correlation_diff < diversity_threshold:
                        diverse = False
                        break
                
                if diverse:
                    selected_pairs.append(pair)
                    used_tickers.add(ticker1)
                    used_tickers.add(ticker2)
            
            self.logger.info(f"Selected {len(selected_pairs)} pairs:")
            for i, pair in enumerate(selected_pairs, 1):
                self.logger.info(f"  {i}. {pair['ticker1']}-{pair['ticker2']}: {pair['correlation']:.3f}")
            
            self.selected_pairs = selected_pairs
            return selected_pairs
            
        except Exception as e:
            self.logger.error(f"Error selecting top pairs: {str(e)}")
            raise
    
    def analyze_pair_stability(self, 
                             returns_data: pd.DataFrame,
                             pairs: List[Dict],
                             window_size: int = 252) -> List[Dict]:
        """
        Analyze correlation stability over time
        
        Args:
            returns_data (pd.DataFrame): Returns data
            pairs (List[Dict]): Selected pairs
            window_size (int): Rolling window size (default: 1 year)
            
        Returns:
            List[Dict]: Pairs with stability metrics
        """
        try:
            self.logger.info(f"Analyzing correlation stability with {window_size}-day rolling window...")
            
            stable_pairs = []
            
            for pair in pairs:
                ticker1, ticker2 = pair['ticker1'], pair['ticker2']
                
                # Get returns for both tickers
                if ticker1 in returns_data.columns and ticker2 in returns_data.columns:
                    returns1 = returns_data[ticker1].dropna()
                    returns2 = returns_data[ticker2].dropna()
                    
                    # Align data
                    aligned_data = pd.concat([returns1, returns2], axis=1).dropna()
                    
                    if len(aligned_data) >= window_size:
                        # Calculate rolling correlation
                        rolling_corr = aligned_data.iloc[:, 0].rolling(window_size).corr(aligned_data.iloc[:, 1])
                        
                        # Calculate stability metrics
                        correlation_std = rolling_corr.std()
                        correlation_range = rolling_corr.max() - rolling_corr.min()
                        correlation_mean = rolling_corr.mean()
                        
                        stability_metrics = {
                            'correlation_std': correlation_std,
                            'correlation_range': correlation_range,
                            'correlation_mean': correlation_mean,
                            'stability_score': 1 / (1 + correlation_std)  # Higher is more stable
                        }
                        
                        pair.update(stability_metrics)
                        stable_pairs.append(pair)
                        
                        self.logger.info(f"{ticker1}-{ticker2}: Mean={correlation_mean:.3f}, Std={correlation_std:.3f}, Stability={stability_metrics['stability_score']:.3f}")
            
            # Sort by stability score
            stable_pairs.sort(key=lambda x: x.get('stability_score', 0), reverse=True)
            
            self.logger.info(f"Analyzed stability for {len(stable_pairs)} pairs")
            
            return stable_pairs
            
        except Exception as e:
            self.logger.error(f"Error analyzing pair stability: {str(e)}")
            raise
    
    def save_results(self, 
                    pairs: List[Dict],
                    filename: str = 'correlation_analysis_results.json') -> None:
        """
        Save correlation analysis results
        
        Args:
            pairs (List[Dict]): Selected pairs with analysis
            filename (str): Output filename
        """
        try:
            import json
            
            output_path = Config.get_results_path(filename)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Convert results
            json_results = json.loads(
                json.dumps(pairs, default=convert_numpy)
            )
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Saved correlation analysis results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def generate_summary_report(self, pairs: List[Dict]) -> Dict:
        """
        Generate summary report of correlation analysis
        
        Args:
            pairs (List[Dict]): Selected pairs
            
        Returns:
            Dict: Summary statistics
        """
        try:
            correlations = [pair['correlation'] for pair in pairs]
            abs_correlations = [pair['abs_correlation'] for pair in pairs]
            
            summary = {
                'total_pairs_analyzed': len(pairs),
                'avg_correlation': np.mean(correlations),
                'avg_abs_correlation': np.mean(abs_correlations),
                'min_correlation': min(correlations),
                'max_correlation': max(correlations),
                'correlation_std': np.std(correlations),
                'positive_correlations': sum(1 for c in correlations if c > 0),
                'negative_correlations': sum(1 for c in correlations if c < 0),
                'high_correlation_pairs': sum(1 for c in abs_correlations if c > 0.8),
                'selected_pairs': [
                    {
                        'ticker1': pair['ticker1'],
                        'ticker2': pair['ticker2'],
                        'correlation': pair['correlation'],
                        'stability_score': pair.get('stability_score', 0)
                    }
                    for pair in pairs[:5]  # Top 5 pairs
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise 