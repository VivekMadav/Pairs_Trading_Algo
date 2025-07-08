"""
Simple Backtesting Framework for Phase 1
Tests data collection and processing components
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import Config

class SimpleBacktest:
    """Simple backtesting framework for testing data components"""
    
    def __init__(self):
        """Initialize SimpleBacktest"""
        self.logger = get_logger('SimpleBacktest', 'simple_backtest.log')
        self.results = {}
    
    def test_data_quality(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Test data quality metrics
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data to test
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        try:
            self.logger.info("Testing data quality...")
            
            quality_metrics = {}
            
            for ticker, data in price_data.items():
                if data.empty:
                    quality_metrics[ticker] = 0.0
                    continue
                
                # Calculate basic quality metrics
                total_days = len(data)
                missing_days = data.isnull().sum().sum()
                zero_prices = ((data[['open', 'high', 'low', 'close']] <= 0).sum().sum())
                zero_volume = (data['volume'] <= 0).sum()
                
                # Calculate quality score
                quality_score = 1.0 - (missing_days + zero_prices + zero_volume) / (total_days * 5)
                quality_score = max(0.0, min(1.0, quality_score))
                
                quality_metrics[ticker] = quality_score
                
                self.logger.info(f"{ticker}: Quality Score = {quality_score:.3f}")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error in test_data_quality: {str(e)}")
            raise
    
    def test_data_completeness(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Test data completeness metrics
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data to test
            
        Returns:
            Dict[str, Dict]: Completeness metrics by ticker
        """
        try:
            self.logger.info("Testing data completeness...")
            
            completeness_metrics = {}
            
            for ticker, data in price_data.items():
                if data.empty:
                    completeness_metrics[ticker] = {
                        'total_days': 0,
                        'expected_days': 0,
                        'completeness_ratio': 0.0,
                        'date_range': None
                    }
                    continue
                
                # Calculate date range
                start_date = data['date'].min()
                end_date = data['date'].max()
                
                # Calculate expected trading days
                expected_days = len(pd.bdate_range(start=start_date, end=end_date))
                actual_days = len(data)
                
                # Calculate completeness ratio
                completeness_ratio = actual_days / expected_days if expected_days > 0 else 0.0
                
                completeness_metrics[ticker] = {
                    'total_days': actual_days,
                    'expected_days': expected_days,
                    'completeness_ratio': completeness_ratio,
                    'date_range': f"{start_date} to {end_date}"
                }
                
                self.logger.info(f"{ticker}: {actual_days}/{expected_days} days ({completeness_ratio:.2%})")
            
            return completeness_metrics
            
        except Exception as e:
            self.logger.error(f"Error in test_data_completeness: {str(e)}")
            raise
    
    def test_data_consistency(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Test data consistency metrics
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data to test
            
        Returns:
            Dict[str, Dict]: Consistency metrics by ticker
        """
        try:
            self.logger.info("Testing data consistency...")
            
            consistency_metrics = {}
            
            for ticker, data in price_data.items():
                if data.empty or len(data) < 2:
                    consistency_metrics[ticker] = {
                        'max_gap': 0,
                        'avg_gap': 0,
                        'consistency_score': 0.0
                    }
                    continue
                
                # Calculate date gaps
                date_diffs = data['date'].diff().dt.days
                max_gap = date_diffs.max()
                avg_gap = date_diffs.mean()
                
                # Calculate consistency score
                consistency_score = 1.0 if max_gap <= Config.MAX_MISSING_DAYS else 0.5
                
                consistency_metrics[ticker] = {
                    'max_gap': max_gap,
                    'avg_gap': avg_gap,
                    'consistency_score': consistency_score
                }
                
                self.logger.info(f"{ticker}: Max gap = {max_gap} days, Consistency = {consistency_score:.2f}")
            
            return consistency_metrics
            
        except Exception as e:
            self.logger.error(f"Error in test_data_consistency: {str(e)}")
            raise
    
    def run_comprehensive_test(self, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run comprehensive backtesting on data
        
        Args:
            price_data (Dict[str, pd.DataFrame]): Price data to test
            
        Returns:
            Dict: Comprehensive test results
        """
        try:
            self.logger.info("Running comprehensive data backtest...")
            
            # Run all tests
            quality_results = self.test_data_quality(price_data)
            completeness_results = self.test_data_completeness(price_data)
            consistency_results = self.test_data_consistency(price_data)
            
            # Compile results
            comprehensive_results = {
                'quality_metrics': quality_results,
                'completeness_metrics': completeness_results,
                'consistency_metrics': consistency_results,
                'summary': self._generate_summary(quality_results, completeness_results, consistency_results)
            }
            
            self.results = comprehensive_results
            
            # Log summary
            self._log_summary(comprehensive_results['summary'])
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error in run_comprehensive_test: {str(e)}")
            raise
    
    def _generate_summary(self, 
                         quality_results: Dict[str, float],
                         completeness_results: Dict[str, Dict],
                         consistency_results: Dict[str, Dict]) -> Dict:
        """
        Generate summary statistics
        
        Args:
            quality_results (Dict[str, float]): Quality test results
            completeness_results (Dict[str, Dict]): Completeness test results
            consistency_results (Dict[str, Dict]): Consistency test results
            
        Returns:
            Dict: Summary statistics
        """
        try:
            tickers = list(quality_results.keys())
            
            # Calculate averages
            avg_quality = np.mean(list(quality_results.values()))
            avg_completeness = np.mean([comp['completeness_ratio'] for comp in completeness_results.values()])
            avg_consistency = np.mean([cons['consistency_score'] for cons in consistency_results.values()])
            
            # Count passing criteria
            quality_passing = sum(1 for score in quality_results.values() if score >= Config.MIN_DATA_QUALITY_SCORE)
            completeness_passing = sum(1 for comp in completeness_results.values() if comp['completeness_ratio'] >= 0.8)
            consistency_passing = sum(1 for cons in consistency_results.values() if cons['consistency_score'] >= 0.8)
            
            summary = {
                'total_tickers': len(tickers),
                'avg_quality_score': avg_quality,
                'avg_completeness_ratio': avg_completeness,
                'avg_consistency_score': avg_consistency,
                'quality_passing': quality_passing,
                'completeness_passing': completeness_passing,
                'consistency_passing': consistency_passing,
                'overall_passing': min(quality_passing, completeness_passing, consistency_passing)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def _log_summary(self, summary: Dict) -> None:
        """
        Log summary results
        
        Args:
            summary (Dict): Summary statistics
        """
        try:
            self.logger.info("=== BACKTEST SUMMARY ===")
            self.logger.info(f"Total tickers: {summary['total_tickers']}")
            self.logger.info(f"Average quality score: {summary['avg_quality_score']:.3f}")
            self.logger.info(f"Average completeness ratio: {summary['avg_completeness_ratio']:.3f}")
            self.logger.info(f"Average consistency score: {summary['avg_consistency_score']:.3f}")
            self.logger.info(f"Quality passing: {summary['quality_passing']}/{summary['total_tickers']}")
            self.logger.info(f"Completeness passing: {summary['completeness_passing']}/{summary['total_tickers']}")
            self.logger.info(f"Consistency passing: {summary['consistency_passing']}/{summary['total_tickers']}")
            self.logger.info(f"Overall passing: {summary['overall_passing']}/{summary['total_tickers']}")
            self.logger.info("========================")
            
        except Exception as e:
            self.logger.error(f"Error logging summary: {str(e)}")
    
    def save_results(self, filename: str = 'backtest_results.json') -> None:
        """
        Save backtest results to file
        
        Args:
            filename (str): Output filename
        """
        try:
            if not self.results:
                self.logger.warning("No results to save")
                return
            
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
                json.dumps(self.results, default=convert_numpy)
            )
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Saved backtest results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise 