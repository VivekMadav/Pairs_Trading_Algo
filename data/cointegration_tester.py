"""
Cointegration Testing for Pairs Trading Model
Handles Engle-Granger, Johansen tests, spread calculation, and threshold optimization
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
from utils.logger import get_logger
from config import Config

class CointegrationTester:
    """Tests cointegration between stock pairs and calculates trading parameters"""
    
    def __init__(self):
        """Initialize CointegrationTester"""
        self.logger = get_logger('CointegrationTester', 'cointegration_tester.log')
        self.price_data = None
        self.test_results = None
        self.spread_data = None
    
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
    
    def load_selected_pairs(self, pairs_path: str = None) -> pd.DataFrame:
        """
        Load selected pairs from Phase 2
        
        Args:
            pairs_path (str): Path to selected pairs file
            
        Returns:
            pd.DataFrame: Selected pairs data
        """
        try:
            if pairs_path is None:
                pairs_path = Config.get_data_path('phase2_selected_pairs.csv')
            
            self.logger.info(f"Loading selected pairs from {pairs_path}")
            
            pairs_data = pd.read_csv(pairs_path)
            
            self.logger.info(f"Loaded {len(pairs_data)} pairs for cointegration testing")
            
            return pairs_data
            
        except Exception as e:
            self.logger.error(f"Error loading selected pairs: {str(e)}")
            raise
    
    def prepare_pair_data(self, 
                         price_data: pd.DataFrame,
                         ticker1: str,
                         ticker2: str) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare price data for a specific pair
        
        Args:
            price_data (pd.DataFrame): Price data
            ticker1 (str): First ticker
            ticker2 (str): Second ticker
            
        Returns:
            Tuple[pd.Series, pd.Series]: Aligned price series for both tickers
        """
        try:
            # Filter data for both tickers
            ticker1_data = price_data[price_data['ticker'] == ticker1].set_index('date')['close']
            ticker2_data = price_data[price_data['ticker'] == ticker2].set_index('date')['close']
            
            # Align data on common dates
            aligned_data = pd.concat([ticker1_data, ticker2_data], axis=1).dropna()
            aligned_data.columns = [ticker1, ticker2]
            
            if len(aligned_data) < 100:  # Minimum data requirement
                raise ValueError(f"Insufficient data for pair {ticker1}-{ticker2}: {len(aligned_data)} observations")
            
            self.logger.info(f"Prepared data for {ticker1}-{ticker2}: {len(aligned_data)} observations")
            
            return aligned_data[ticker1], aligned_data[ticker2]
            
        except Exception as e:
            self.logger.error(f"Error preparing data for {ticker1}-{ticker2}: {str(e)}")
            raise
    
    def engle_granger_test(self, 
                          series1: pd.Series,
                          series2: pd.Series,
                          ticker1: str,
                          ticker2: str) -> Dict:
        """
        Perform Engle-Granger cointegration test
        
        Args:
            series1 (pd.Series): First price series
            series2 (pd.Series): Second price series
            ticker1 (str): First ticker name
            ticker2 (str): Second ticker name
            
        Returns:
            Dict: Test results
        """
        try:
            self.logger.info(f"Performing Engle-Granger test for {ticker1}-{ticker2}")
            
            # Perform cointegration test
            result = coint(series1, series2)
            score = result[0]
            pvalue = result[1]
            critical_values = result[2]
            
            # Calculate hedge ratio using OLS
            model = sm.OLS(series1, sm.add_constant(series2))
            results = model.fit()
            hedge_ratio = results.params.iloc[1]
            intercept = results.params.iloc[0]
            
            # Calculate spread
            spread = series1 - (hedge_ratio * series2 + intercept)
            
            # Test spread for stationarity
            adf_result = adfuller(spread.dropna())
            adf_stat = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical = adf_result[4]
            
            # Determine if cointegrated
            is_cointegrated = pvalue < 0.05 and adf_pvalue < 0.05
            
            results = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'engle_granger_score': score,
                'engle_granger_pvalue': pvalue,
                'engle_granger_critical_values': critical_values,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'spread_adf_stat': adf_stat,
                'spread_adf_pvalue': adf_pvalue,
                'spread_adf_critical': adf_critical,
                'is_cointegrated': is_cointegrated,
                'spread_series': spread
            }
            
            self.logger.info(f"Engle-Granger test for {ticker1}-{ticker2}: p-value={pvalue:.4f}, cointegrated={is_cointegrated}")
            
            return results
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in Engle-Granger test for {ticker1}-{ticker2}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def johansen_test(self, 
                     series1: pd.Series,
                     series2: pd.Series,
                     ticker1: str,
                     ticker2: str) -> Dict:
        """
        Perform Johansen cointegration test
        
        Args:
            series1 (pd.Series): First price series
            series2 (pd.Series): Second price series
            ticker1 (str): First ticker name
            ticker2 (str): Second ticker name
            
        Returns:
            Dict: Test results
        """
        try:
            self.logger.info(f"Performing Johansen test for {ticker1}-{ticker2}")
            
            # Prepare data for Johansen test
            data = pd.concat([series1, series2], axis=1).dropna()
            
            # Perform Johansen test
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            # Extract test statistics
            trace_stat = result.lr1[0]  # Trace statistic
            max_eigen_stat = result.lr2[0]  # Max eigenvalue statistic
            
            # Critical values
            trace_critical = result.cvt[:, 0]
            max_eigen_critical = result.cvt[:, 1]
            
            # Determine cointegration rank
            trace_cointegrated = trace_stat > trace_critical[1]  # 95% confidence
            max_eigen_cointegrated = max_eigen_stat > max_eigen_critical[1]  # 95% confidence
            
            is_cointegrated = trace_cointegrated and max_eigen_cointegrated
            
            results = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'johansen_trace_stat': trace_stat,
                'johansen_max_eigen_stat': max_eigen_stat,
                'johansen_trace_critical': trace_critical,
                'johansen_max_eigen_critical': max_eigen_critical,
                'johansen_trace_cointegrated': trace_cointegrated,
                'johansen_max_eigen_cointegrated': max_eigen_cointegrated,
                'johansen_is_cointegrated': is_cointegrated
            }
            
            self.logger.info(f"Johansen test for {ticker1}-{ticker2}: trace={trace_stat:.4f}, max_eigen={max_eigen_stat:.4f}, cointegrated={is_cointegrated}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Johansen test for {ticker1}-{ticker2}: {str(e)}")
            raise
    
    def calculate_spread_statistics(self, 
                                  spread: pd.Series,
                                  ticker1: str,
                                  ticker2: str) -> Dict:
        """
        Calculate comprehensive spread statistics
        
        Args:
            spread (pd.Series): Spread series
            ticker1 (str): First ticker name
            ticker2 (str): Second ticker name
            
        Returns:
            Dict: Spread statistics
        """
        try:
            self.logger.info(f"Calculating spread statistics for {ticker1}-{ticker2}")
            
            # Remove NaN values
            clean_spread = spread.dropna()
            
            # Basic statistics
            mean_spread = clean_spread.mean()
            std_spread = clean_spread.std()
            min_spread = clean_spread.min()
            max_spread = clean_spread.max()
            
            # Calculate z-score
            z_score = (clean_spread - mean_spread) / std_spread
            
            # Percentiles
            percentiles = np.percentile(clean_spread, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Hurst exponent (measure of mean reversion)
            hurst_exponent = self._calculate_hurst_exponent(clean_spread)
            
            # Half-life (time for spread to revert to mean)
            half_life = self._calculate_half_life(clean_spread)
            
            # Variance ratio test
            variance_ratio = self._calculate_variance_ratio(clean_spread)
            
            # Jarque-Bera test for normality
            jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(clean_spread)
            
            statistics = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'mean_spread': mean_spread,
                'std_spread': std_spread,
                'min_spread': min_spread,
                'max_spread': max_spread,
                'spread_range': max_spread - min_spread,
                'percentiles': percentiles.tolist(),
                'hurst_exponent': hurst_exponent,
                'half_life': half_life,
                'variance_ratio': variance_ratio,
                'jarque_bera_stat': jarque_bera_stat,
                'jarque_bera_pvalue': jarque_bera_pvalue,
                'is_normal': jarque_bera_pvalue > 0.05,
                'z_score_series': z_score
            }
            
            self.logger.info(f"Spread statistics for {ticker1}-{ticker2}: mean={mean_spread:.4f}, std={std_spread:.4f}, half-life={half_life:.1f} days")
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating spread statistics for {ticker1}-{ticker2}: {str(e)}")
            raise
    
    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent to test for mean reversion
        
        Args:
            series (pd.Series): Time series
            
        Returns:
            float: Hurst exponent
        """
        try:
            # Calculate returns
            returns = series.diff().dropna()
            
            # Calculate variance of returns
            var_returns = returns.var()
            
            # Calculate variance of cumulative returns
            cum_returns = returns.cumsum()
            var_cum = cum_returns.var()
            
            # Calculate Hurst exponent
            hurst = 0.5 * np.log(var_cum / var_returns) / np.log(len(returns))
            
            return hurst
            
        except Exception:
            return 0.5  # Default to random walk
    
    def _calculate_half_life(self, series: pd.Series) -> float:
        """
        Calculate half-life of mean reversion
        
        Args:
            series (pd.Series): Time series
            
        Returns:
            float: Half-life in periods
        """
        try:
            # Calculate spread change
            spread_lag = series.shift(1)
            spread_ret = series - spread_lag
            
            # Remove NaN values
            valid_data = pd.concat([spread_lag, spread_ret], axis=1).dropna()
            
            # OLS regression
            model = sm.OLS(valid_data.iloc[:, 1], sm.add_constant(valid_data.iloc[:, 0]))
            results = model.fit()
            
            # Calculate half-life
            half_life = -np.log(2) / results.params[1]
            
            return max(0, half_life)  # Ensure non-negative
            
        except Exception:
            return 252  # Default to 1 year
    
    def _calculate_variance_ratio(self, series: pd.Series) -> float:
        """
        Calculate variance ratio test statistic
        
        Args:
            series (pd.Series): Time series
            
        Returns:
            float: Variance ratio
        """
        try:
            # Calculate returns
            returns = series.diff().dropna()
            
            # Calculate variances for different periods
            var_1 = returns.var()
            var_2 = returns.rolling(2).sum().var()
            
            # Variance ratio
            variance_ratio = var_2 / (2 * var_1)
            
            return variance_ratio
            
        except Exception:
            return 1.0  # Default to random walk
    
    def optimize_thresholds(self, 
                           z_score: pd.Series,
                           ticker1: str,
                           ticker2: str) -> Dict:
        """
        Optimize entry and exit thresholds based on z-score distribution
        
        Args:
            z_score (pd.Series): Z-score series
            ticker1 (str): First ticker name
            ticker2 (str): Second ticker name
            
        Returns:
            Dict: Optimized thresholds
        """
        try:
            self.logger.info(f"Optimizing thresholds for {ticker1}-{ticker2}")
            
            clean_z_score = z_score.dropna()
            
            # Calculate percentiles for different threshold levels
            percentiles = [80, 85, 90, 95, 99]
            thresholds = {}
            
            for p in percentiles:
                threshold = np.percentile(np.abs(clean_z_score), p)
                thresholds[f'p{p}'] = threshold
            
            # Calculate optimal thresholds based on historical performance
            # Entry threshold: 95th percentile
            entry_threshold = thresholds['p95']
            
            # Exit threshold: 50th percentile (mean reversion)
            exit_threshold = np.percentile(np.abs(clean_z_score), 50)
            
            # Stop loss threshold: 99th percentile
            stop_loss_threshold = thresholds['p99']
            
            # Calculate expected holding period
            holding_period = self._calculate_expected_holding_period(clean_z_score, entry_threshold)
            
            thresholds_result = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'stop_loss_threshold': stop_loss_threshold,
                'percentile_thresholds': thresholds,
                'expected_holding_period': holding_period
            }
            
            self.logger.info(f"Thresholds for {ticker1}-{ticker2}: entry={entry_threshold:.2f}, exit={exit_threshold:.2f}, stop_loss={stop_loss_threshold:.2f}")
            
            return thresholds_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing thresholds for {ticker1}-{ticker2}: {str(e)}")
            raise
    
    def _calculate_expected_holding_period(self, 
                                         z_score: pd.Series,
                                         entry_threshold: float) -> float:
        """
        Calculate expected holding period for trades
        
        Args:
            z_score (pd.Series): Z-score series
            entry_threshold (float): Entry threshold
            
        Returns:
            float: Expected holding period in days
        """
        try:
            # Find periods where z-score exceeds entry threshold
            entry_signals = np.abs(z_score) > entry_threshold
            
            # Calculate average time to return to mean
            holding_periods = []
            current_period = 0
            
            for i, signal in enumerate(entry_signals):
                if signal:
                    current_period += 1
                elif current_period > 0:
                    holding_periods.append(current_period)
                    current_period = 0
            
            if holding_periods:
                return np.mean(holding_periods)
            else:
                return 10  # Default holding period
            
        except Exception:
            return 10  # Default holding period
    
    def test_pair_stability(self, 
                           spread: pd.Series,
                           ticker1: str,
                           ticker2: str,
                           window_size: int = 126) -> Dict:
        """
        Test pair stability over time using rolling windows
        
        Args:
            spread (pd.Series): Spread series
            ticker1 (str): First ticker name
            ticker2 (str): Second ticker name
            window_size (int): Rolling window size (6 months)
            
        Returns:
            Dict: Stability test results
        """
        try:
            self.logger.info(f"Testing pair stability for {ticker1}-{ticker2}")
            
            clean_spread = spread.dropna()
            
            if len(clean_spread) < window_size * 2:
                raise ValueError(f"Insufficient data for stability test: {len(clean_spread)} observations")
            
            # Calculate rolling statistics
            rolling_mean = clean_spread.rolling(window_size).mean()
            rolling_std = clean_spread.rolling(window_size).std()
            
            # Calculate stability metrics
            mean_stability = 1 / (1 + rolling_mean.std())
            std_stability = 1 / (1 + rolling_std.std())
            
            # Calculate structural breaks
            structural_breaks = self._detect_structural_breaks(clean_spread)
            
            # Calculate regime changes
            regime_changes = self._detect_regime_changes(clean_spread)
            
            stability_results = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'mean_stability': mean_stability,
                'std_stability': std_stability,
                'overall_stability': (mean_stability + std_stability) / 2,
                'structural_breaks': structural_breaks,
                'regime_changes': regime_changes,
                'is_stable': (mean_stability + std_stability) / 2 > 0.8
            }
            
            self.logger.info(f"Stability test for {ticker1}-{ticker2}: overall_stability={stability_results['overall_stability']:.3f}")
            
            return stability_results
            
        except Exception as e:
            self.logger.error(f"Error testing pair stability for {ticker1}-{ticker2}: {str(e)}")
            raise
    
    def _detect_structural_breaks(self, series: pd.Series) -> int:
        """
        Detect structural breaks in the series
        
        Args:
            series (pd.Series): Time series
            
        Returns:
            int: Number of structural breaks
        """
        try:
            # Simple structural break detection using rolling mean changes
            rolling_mean = series.rolling(20).mean()
            mean_changes = np.abs(rolling_mean.diff()) > 2 * series.std()
            
            return int(mean_changes.sum())
            
        except Exception:
            return 0
    
    def _detect_regime_changes(self, series: pd.Series) -> int:
        """
        Detect regime changes in the series
        
        Args:
            series (pd.Series): Time series
            
        Returns:
            int: Number of regime changes
        """
        try:
            # Simple regime change detection using volatility clustering
            returns = series.diff().dropna()
            volatility = returns.rolling(20).std()
            high_vol_periods = volatility > volatility.quantile(0.8)
            
            return int(high_vol_periods.sum())
            
        except Exception:
            return 0
    
    def run_complete_analysis(self, pairs_data: pd.DataFrame) -> List[Dict]:
        """
        Run complete cointegration analysis for all pairs
        
        Args:
            pairs_data (pd.DataFrame): Selected pairs from Phase 2
            
        Returns:
            List[Dict]: Complete analysis results
        """
        try:
            self.logger.info("Starting complete cointegration analysis...")
            
            if self.price_data is None:
                raise ValueError("Price data not loaded. Call load_price_data() first.")
            
            results = []
            
            for _, pair in pairs_data.iterrows():
                ticker1, ticker2 = pair['ticker1'], pair['ticker2']
                
                try:
                    self.logger.info(f"Analyzing pair: {ticker1}-{ticker2}")
                    
                    # Prepare data
                    series1, series2 = self.prepare_pair_data(self.price_data, ticker1, ticker2)
                    
                    # Engle-Granger test
                    eg_results = self.engle_granger_test(series1, series2, ticker1, ticker2)
                    
                    # Johansen test
                    johansen_results = self.johansen_test(series1, series2, ticker1, ticker2)
                    
                    # Spread statistics
                    spread_stats = self.calculate_spread_statistics(eg_results['spread_series'], ticker1, ticker2)
                    
                    # Optimize thresholds
                    thresholds = self.optimize_thresholds(spread_stats['z_score_series'], ticker1, ticker2)
                    
                    # Test stability
                    stability = self.test_pair_stability(eg_results['spread_series'], ticker1, ticker2)
                    
                    # Combine all results
                    pair_result = {
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'correlation': pair['correlation'],
                        'stability_score': pair['stability_score'],
                        'engle_granger': eg_results,
                        'johansen': johansen_results,
                        'spread_statistics': spread_stats,
                        'thresholds': thresholds,
                        'stability_test': stability,
                        'is_tradeable': (
                            eg_results['is_cointegrated'] and 
                            johansen_results['johansen_is_cointegrated'] and
                            stability['is_stable'] and
                            spread_stats['half_life'] < 252  # Less than 1 year
                        )
                    }
                    
                    results.append(pair_result)
                    
                    self.logger.info(f"Completed analysis for {ticker1}-{ticker2}: tradeable={pair_result['is_tradeable']}")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing pair {ticker1}-{ticker2}: {str(e)}")
                    continue
            
            self.test_results = results
            self.logger.info(f"Completed analysis for {len(results)} pairs")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {str(e)}")
            raise
    
    def save_results(self, 
                    results: List[Dict],
                    filename: str = 'cointegration_test_results.json') -> None:
        """
        Save cointegration test results
        
        Args:
            results (List[Dict]): Test results
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
                    'stability_score': float(result['stability_score']),
                    'is_tradeable': bool(result['is_tradeable']),
                    'engle_granger': {
                        'pvalue': float(result['engle_granger']['engle_granger_pvalue']),
                        'is_cointegrated': bool(result['engle_granger']['is_cointegrated'])
                    },
                    'johansen': {
                        'is_cointegrated': bool(result['johansen']['johansen_is_cointegrated'])
                    },
                    'spread_statistics': {
                        'half_life': float(result['spread_statistics']['half_life']),
                        'hurst_exponent': float(result['spread_statistics']['hurst_exponent']),
                        'mean_spread': float(result['spread_statistics']['mean_spread']),
                        'std_spread': float(result['spread_statistics']['std_spread'])
                    },
                    'thresholds': {
                        'entry_threshold': float(result['thresholds']['entry_threshold']),
                        'exit_threshold': float(result['thresholds']['exit_threshold']),
                        'stop_loss_threshold': float(result['thresholds']['stop_loss_threshold'])
                    },
                    'stability_test': {
                        'overall_stability': float(result['stability_test']['overall_stability'])
                    }
                }
                simplified_results.append(simplified_result)
            
            with open(output_path, 'w') as f:
                json.dump(simplified_results, f, indent=2)
            
            self.logger.info(f"Saved cointegration test results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """
        Generate summary report of cointegration analysis
        
        Args:
            results (List[Dict]): Test results
            
        Returns:
            Dict: Summary statistics
        """
        try:
            tradeable_pairs = [r for r in results if r['is_tradeable']]
            
            summary = {
                'total_pairs_tested': len(results),
                'tradeable_pairs': len(tradeable_pairs),
                'tradeable_ratio': len(tradeable_pairs) / len(results) if results else 0,
                'avg_half_life': np.mean([r['spread_statistics']['half_life'] for r in results]),
                'avg_hurst_exponent': np.mean([r['spread_statistics']['hurst_exponent'] for r in results]),
                'avg_stability': np.mean([r['stability_test']['overall_stability'] for r in results]),
                'tradeable_pairs_details': [
                    {
                        'ticker1': pair['ticker1'],
                        'ticker2': pair['ticker2'],
                        'correlation': pair['correlation'],
                        'half_life': pair['spread_statistics']['half_life'],
                        'hurst_exponent': pair['spread_statistics']['hurst_exponent'],
                        'stability': pair['stability_test']['overall_stability'],
                        'entry_threshold': pair['thresholds']['entry_threshold'],
                        'exit_threshold': pair['thresholds']['exit_threshold']
                    }
                    for pair in tradeable_pairs
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise 