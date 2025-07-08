"""
Backtesting Framework for Pairs Trading Model
Handles performance calculation, risk metrics, and comprehensive reporting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import get_logger
from config import Config

class Backtester:
    """Backtesting framework for pairs trading strategies"""
    
    def __init__(self):
        """Initialize Backtester"""
        self.logger = get_logger('Backtester', 'backtester.log')
        self.results = None
        self.performance_metrics = None
    
    def calculate_performance_metrics(self, 
                                    returns: pd.Series,
                                    benchmark_returns: pd.Series = None) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns (pd.Series): Strategy returns
            benchmark_returns (pd.Series): Benchmark returns (optional)
            
        Returns:
            Dict: Performance metrics
        """
        try:
            # Remove NaN values
            clean_returns = returns.dropna()
            
            if len(clean_returns) == 0:
                raise ValueError("No valid returns data")
            
            # Basic metrics
            total_return = (1 + clean_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(clean_returns)) - 1
            volatility = clean_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(clean_returns)
            var_95 = np.percentile(clean_returns, 5)
            cvar_95 = clean_returns[clean_returns <= var_95].mean()
            
            # Win/loss metrics
            winning_days = (clean_returns > 0).sum()
            losing_days = (clean_returns < 0).sum()
            total_days = len(clean_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            avg_win = clean_returns[clean_returns > 0].mean() if winning_days > 0 else 0
            avg_loss = clean_returns[clean_returns < 0].mean() if losing_days > 0 else 0
            profit_factor = abs(avg_win * winning_days / (avg_loss * losing_days)) if losing_days > 0 and avg_loss != 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio
            downside_returns = clean_returns[clean_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Benchmark comparison
            benchmark_metrics = {}
            if benchmark_returns is not None:
                benchmark_clean = benchmark_returns.dropna()
                if len(benchmark_clean) > 0:
                    benchmark_total_return = (1 + benchmark_clean).prod() - 1
                    benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_clean)) - 1
                    benchmark_volatility = benchmark_clean.std() * np.sqrt(252)
                    
                    benchmark_metrics = {
                        'benchmark_total_return': benchmark_total_return,
                        'benchmark_annualized_return': benchmark_annualized,
                        'benchmark_volatility': benchmark_volatility,
                        'excess_return': annualized_return - benchmark_annualized,
                        'information_ratio': (annualized_return - benchmark_annualized) / volatility if volatility > 0 else 0
                    }
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'winning_days': winning_days,
                'losing_days': losing_days,
                'total_days': total_days,
                **benchmark_metrics
            }
            
            self.logger.info(f"Calculated performance metrics: Sharpe={sharpe_ratio:.3f}, MaxDD={max_drawdown:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            returns (pd.Series): Returns series
            
        Returns:
            float: Maximum drawdown
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def calculate_trade_metrics(self, 
                              signals: pd.Series,
                              returns: pd.Series) -> Dict:
        """
        Calculate trade-specific metrics
        
        Args:
            signals (pd.Series): Trading signals
            returns (pd.Series): Strategy returns
            
        Returns:
            Dict: Trade metrics
        """
        try:
            # Find trade entry and exit points
            trade_entries = signals.diff() != 0
            trade_exits = signals.diff() != 0
            
            # Calculate trade durations
            trade_durations = []
            trade_returns = []
            current_trade_start = None
            current_position = 0
            
            for i, (signal, ret) in enumerate(zip(signals, returns)):
                if pd.isna(signal) or pd.isna(ret):
                    continue
                
                # New trade starts
                if signal != current_position and signal != 0:
                    if current_trade_start is not None:
                        # Close previous trade
                        duration = i - current_trade_start
                        trade_durations.append(duration)
                    
                    current_trade_start = i
                    current_position = signal
                
                # Trade ends
                elif signal == 0 and current_position != 0:
                    if current_trade_start is not None:
                        duration = i - current_trade_start
                        trade_durations.append(duration)
                        current_trade_start = None
                        current_position = 0
            
            # Calculate trade returns
            cumulative_returns = (1 + returns).cumprod()
            trade_returns = []
            
            for i in range(len(signals) - 1):
                if signals.iloc[i] != 0 and signals.iloc[i+1] == 0:
                    # Trade exit
                    if i > 0:
                        trade_return = (cumulative_returns.iloc[i] / cumulative_returns.iloc[i-1]) - 1
                        trade_returns.append(trade_return)
            
            metrics = {
                'total_trades': len(trade_returns),
                'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
                'avg_trade_return': np.mean(trade_returns) if trade_returns else 0,
                'trade_return_std': np.std(trade_returns) if trade_returns else 0,
                'best_trade': max(trade_returns) if trade_returns else 0,
                'worst_trade': min(trade_returns) if trade_returns else 0,
                'profitable_trades': sum(1 for r in trade_returns if r > 0) if trade_returns else 0,
                'losing_trades': sum(1 for r in trade_returns if r < 0) if trade_returns else 0
            }
            
            if metrics['total_trades'] > 0:
                metrics['trade_win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
            else:
                metrics['trade_win_rate'] = 0
            
            self.logger.info(f"Calculated trade metrics: {metrics['total_trades']} trades")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {str(e)}")
            raise
    
    def run_backtest(self, 
                    signal_results: List[Dict],
                    transaction_cost: float = 0.001,
                    benchmark_returns: pd.Series = None) -> Dict:
        """
        Run comprehensive backtest
        
        Args:
            signal_results (List[Dict]): Signal generation results
            transaction_cost (float): Transaction cost per trade
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            Dict: Backtest results
        """
        try:
            self.logger.info("Running comprehensive backtest...")
            
            all_results = []
            
            for result in signal_results:
                ticker1, ticker2 = result['ticker1'], result['ticker2']
                
                try:
                    self.logger.info(f"Backtesting {ticker1}-{ticker2}")
                    
                    # Get strategy returns
                    strategy_returns = result['strategy_returns']
                    signals = result['signals']
                    
                    # Apply transaction costs
                    if transaction_cost > 0:
                        # Calculate transaction costs
                        position_changes = signals.diff().abs()
                        transaction_costs = position_changes * transaction_cost
                        net_returns = strategy_returns - transaction_costs
                    else:
                        net_returns = strategy_returns
                    
                    # Calculate performance metrics
                    performance_metrics = self.calculate_performance_metrics(
                        net_returns, 
                        benchmark_returns
                    )
                    
                    # Calculate trade metrics
                    trade_metrics = self.calculate_trade_metrics(signals, net_returns)
                    
                    # Combine results
                    pair_result = {
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'correlation': result['correlation'],
                        'performance_metrics': performance_metrics,
                        'trade_metrics': trade_metrics,
                        'returns': net_returns,
                        'signals': signals,
                        'transaction_cost': transaction_cost
                    }
                    
                    all_results.append(pair_result)
                    
                    self.logger.info(f"Completed backtest for {ticker1}-{ticker2}")
                    
                except Exception as e:
                    self.logger.error(f"Error backtesting {ticker1}-{ticker2}: {str(e)}")
                    continue
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(all_results)
            
            backtest_results = {
                'pair_results': all_results,
                'portfolio_metrics': portfolio_metrics,
                'transaction_cost': transaction_cost
            }
            
            self.results = backtest_results
            self.logger.info(f"Completed backtest for {len(all_results)} pairs")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _calculate_portfolio_metrics(self, pair_results: List[Dict]) -> Dict:
        """
        Calculate portfolio-level metrics
        
        Args:
            pair_results (List[Dict]): Individual pair results
            
        Returns:
            Dict: Portfolio metrics
        """
        try:
            # Combine all returns
            all_returns = []
            for result in pair_results:
                returns = result['returns'].dropna()
                all_returns.append(returns)
            
            if not all_returns:
                return {}
            
            # Align all return series
            aligned_returns = pd.concat(all_returns, axis=1).fillna(0)
            
            # Calculate equal-weighted portfolio returns
            portfolio_returns = aligned_returns.mean(axis=1)
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_performance_metrics(portfolio_returns)
            
            # Add portfolio-specific metrics
            portfolio_metrics['num_pairs'] = len(pair_results)
            portfolio_metrics['avg_pair_sharpe'] = np.mean([
                r['performance_metrics']['sharpe_ratio'] for r in pair_results
            ])
            portfolio_metrics['avg_pair_return'] = np.mean([
                r['performance_metrics']['annualized_return'] for r in pair_results
            ])
            
            return portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def generate_report(self, 
                       backtest_results: Dict,
                       output_dir: str = None) -> Dict:
        """
        Generate comprehensive backtest report
        
        Args:
            backtest_results (Dict): Backtest results
            output_dir (str): Output directory
            
        Returns:
            Dict: Report summary
        """
        try:
            if output_dir is None:
                output_dir = Config.get_results_path('')
            
            self.logger.info("Generating comprehensive backtest report...")
            
            pair_results = backtest_results['pair_results']
            portfolio_metrics = backtest_results['portfolio_metrics']
            
            # Create summary report
            summary = {
                'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_pairs': len(pair_results),
                'transaction_cost': backtest_results['transaction_cost'],
                'portfolio_metrics': portfolio_metrics,
                'pair_summaries': []
            }
            
            # Individual pair summaries
            for result in pair_results:
                pair_summary = {
                    'pair': f"{result['ticker1']}-{result['ticker2']}",
                    'correlation': result['correlation'],
                    'annualized_return': result['performance_metrics']['annualized_return'],
                    'sharpe_ratio': result['performance_metrics']['sharpe_ratio'],
                    'max_drawdown': result['performance_metrics']['max_drawdown'],
                    'total_trades': result['trade_metrics']['total_trades'],
                    'win_rate': result['trade_metrics']['trade_win_rate']
                }
                summary['pair_summaries'].append(pair_summary)
            
            # Save detailed report
            report_path = f"{output_dir}/backtest_report.json"
            import json
            
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_summary = json.loads(json.dumps(summary, default=convert_numpy))
            
            with open(report_path, 'w') as f:
                json.dump(json_summary, f, indent=2)
            
            self.logger.info(f"Saved backtest report to {report_path}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
    
    def save_results(self, 
                    backtest_results: Dict,
                    filename: str = 'backtest_results.json') -> None:
        """
        Save backtest results
        
        Args:
            backtest_results (Dict): Backtest results
            filename (str): Output filename
        """
        try:
            import json
            
            output_path = Config.get_results_path(filename)
            
            # Create simplified results structure
            simplified_results = {
                'portfolio_metrics': backtest_results['portfolio_metrics'],
                'transaction_cost': backtest_results['transaction_cost'],
                'pair_results': []
            }
            
            for result in backtest_results['pair_results']:
                simplified_pair = {
                    'ticker1': result['ticker1'],
                    'ticker2': result['ticker2'],
                    'correlation': float(result['correlation']),
                    'performance_metrics': result['performance_metrics'],
                    'trade_metrics': result['trade_metrics']
                }
                simplified_results['pair_results'].append(simplified_pair)
            
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json_results = convert_numpy(simplified_results)
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Saved backtest results to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise 