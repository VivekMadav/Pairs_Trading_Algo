"""
Phase 4: Signal Generation and Backtesting
Main execution script for generating trading signals and running comprehensive backtests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data.signal_generator import SignalGenerator
from data.backtester import Backtester
from utils.logger import get_logger
from config import Config

def main():
    """Main execution function for Phase 4"""
    
    # Setup logging
    logger = get_logger('Phase4', 'phase4.log')
    logger.info("=" * 60)
    logger.info("PHASE 4: SIGNAL GENERATION AND BACKTESTING")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        signal_generator = SignalGenerator()
        backtester = Backtester()
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        
        # Load price data from Phase 1
        price_data = signal_generator.load_price_data()
        logger.info(f"✓ Loaded price data for {price_data['ticker'].nunique()} tickers")
        
        # Load pairs data from Phase 2
        pairs_data = signal_generator.load_pairs_data()
        logger.info(f"✓ Loaded {len(pairs_data)} pairs for analysis")
        
        # Display pairs summary
        logger.info("\nPairs Summary:")
        for _, pair in pairs_data.iterrows():
            logger.info(f"  {pair['ticker1']}-{pair['ticker2']}: {pair['correlation']:.3f}")
        
        # Step 2: Generate signals
        logger.info("\nStep 2: Generating trading signals...")
        
        # Signal generation parameters
        entry_threshold = 2.0      # Z-score threshold for entry
        exit_threshold = 0.5       # Z-score threshold for exit
        stop_loss_threshold = 4.0  # Z-score threshold for stop loss
        
        logger.info(f"Signal Parameters:")
        logger.info(f"  Entry threshold: {entry_threshold}")
        logger.info(f"  Exit threshold: {exit_threshold}")
        logger.info(f"  Stop loss threshold: {stop_loss_threshold}")
        
        # Generate signals for all pairs
        signal_results = signal_generator.generate_all_signals(
            pairs_data,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss_threshold
        )
        
        logger.info(f"✓ Generated signals for {len(signal_results)} pairs")
        
        # Step 3: Run backtesting
        logger.info("\nStep 3: Running backtesting...")
        
        # Backtesting parameters
        transaction_cost = 0.001  # 0.1% transaction cost per trade
        
        logger.info(f"Backtesting Parameters:")
        logger.info(f"  Transaction cost: {transaction_cost:.3%}")
        
        # Run backtest
        backtest_results = backtester.run_backtest(
            signal_results,
            transaction_cost=transaction_cost
        )
        
        logger.info(f"✓ Completed backtesting for {len(backtest_results['pair_results'])} pairs")
        
        # Step 4: Generate comprehensive report
        logger.info("\nStep 4: Generating comprehensive report...")
        
        report_summary = backtester.generate_report(backtest_results)
        
        # Display portfolio summary
        portfolio_metrics = report_summary['portfolio_metrics']
        logger.info("\nPortfolio Performance Summary:")
        logger.info(f"  Number of pairs: {portfolio_metrics.get('num_pairs', 0)}")
        logger.info(f"  Annualized Return: {portfolio_metrics.get('annualized_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Volatility: {portfolio_metrics.get('volatility', 0):.2%}")
        logger.info(f"  Win Rate: {portfolio_metrics.get('win_rate', 0):.2%}")
        
        # Display individual pair results
        logger.info("\nIndividual Pair Results:")
        for pair_summary in report_summary['pair_summaries']:
            logger.info(f"  {pair_summary['pair']}:")
            logger.info(f"    Correlation: {pair_summary['correlation']:.3f}")
            logger.info(f"    Annual Return: {pair_summary['annualized_return']:.2%}")
            logger.info(f"    Sharpe Ratio: {pair_summary['sharpe_ratio']:.3f}")
            logger.info(f"    Max Drawdown: {pair_summary['max_drawdown']:.2%}")
            logger.info(f"    Total Trades: {pair_summary['total_trades']}")
            logger.info(f"    Win Rate: {pair_summary['win_rate']:.2%}")
        
        # Step 5: Save results
        logger.info("\nStep 5: Saving results...")
        
        # Save signal generation results
        signal_generator.save_results(signal_results, 'phase4_signal_results.json')
        logger.info("✓ Saved signal generation results")
        
        # Save backtest results
        backtester.save_results(backtest_results, 'phase4_backtest_results.json')
        logger.info("✓ Saved backtest results")
        
        # Step 6: Generate visualizations
        logger.info("\nStep 6: Generating visualizations...")
        
        generate_visualizations(backtest_results, signal_results)
        logger.info("✓ Generated visualizations")
        
        # Step 7: Summary and recommendations
        logger.info("\nStep 7: Summary and recommendations...")
        
        generate_summary_and_recommendations(report_summary, signal_results)
        
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4 COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Phase 4: {str(e)}")
        raise

def generate_visualizations(backtest_results, signal_results):
    """Generate comprehensive visualizations"""
    
    try:
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pairs Trading Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Performance
        ax1 = axes[0, 0]
        portfolio_returns = []
        dates = []
        
        for result in backtest_results['pair_results']:
            returns = result['returns'].dropna()
            portfolio_returns.append(returns)
            if not dates:
                dates = returns.index
        
        if portfolio_returns:
            # Align all returns
            aligned_returns = pd.concat(portfolio_returns, axis=1).fillna(0)
            portfolio_returns_series = aligned_returns.mean(axis=1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns_series).cumprod()
            
            ax1.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
            ax1.set_title('Portfolio Cumulative Returns')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Return')
            ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        pairs = []
        sharpe_ratios = []
        
        for result in backtest_results['pair_results']:
            pair_name = f"{result['ticker1']}-{result['ticker2']}"
            sharpe = result['performance_metrics']['sharpe_ratio']
            pairs.append(pair_name)
            sharpe_ratios.append(sharpe)
        
        if sharpe_ratios:
            bars = ax2.bar(pairs, sharpe_ratios, alpha=0.7)
            ax2.set_title('Sharpe Ratio by Pair')
            ax2.set_xlabel('Pairs')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Color bars based on performance
            for bar, sharpe in zip(bars, sharpe_ratios):
                if sharpe > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
        
        # 3. Max Drawdown Comparison
        ax3 = axes[1, 0]
        max_drawdowns = []
        
        for result in backtest_results['pair_results']:
            max_dd = result['performance_metrics']['max_drawdown']
            max_drawdowns.append(max_dd)
        
        if max_drawdowns:
            bars = ax3.bar(pairs, max_drawdowns, alpha=0.7, color='orange')
            ax3.set_title('Maximum Drawdown by Pair')
            ax3.set_xlabel('Pairs')
            ax3.set_ylabel('Max Drawdown')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Trade Count and Win Rate
        ax4 = axes[1, 1]
        trade_counts = []
        win_rates = []
        
        for result in backtest_results['pair_results']:
            trade_count = result['trade_metrics']['total_trades']
            win_rate = result['trade_metrics']['trade_win_rate']
            trade_counts.append(trade_count)
            win_rates.append(win_rate)
        
        if trade_counts and win_rates:
            # Create scatter plot
            scatter = ax4.scatter(trade_counts, win_rates, s=100, alpha=0.7)
            ax4.set_title('Trade Count vs Win Rate')
            ax4.set_xlabel('Number of Trades')
            ax4.set_ylabel('Win Rate')
            ax4.grid(True, alpha=0.3)
            
            # Add pair labels
            for i, pair in enumerate(pairs):
                ax4.annotate(pair, (trade_counts[i], win_rates[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Config.get_results_path('phase4_performance_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate additional plots
        generate_signal_analysis_plots(signal_results)
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

def generate_signal_analysis_plots(signal_results):
    """Generate signal analysis plots"""
    
    try:
        # Create figure for signal analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Signal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Z-score distribution
        ax1 = axes[0, 0]
        all_z_scores = []
        
        for result in signal_results:
            z_scores = result['z_score'].dropna()
            all_z_scores.extend(z_scores.values)
        
        if all_z_scores:
            ax1.hist(all_z_scores, bins=50, alpha=0.7, density=True)
            ax1.set_title('Z-Score Distribution')
            ax1.set_xlabel('Z-Score')
            ax1.set_ylabel('Density')
            ax1.grid(True, alpha=0.3)
            
            # Add vertical lines for thresholds
            ax1.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Entry Threshold')
            ax1.axvline(x=-2.0, color='red', linestyle='--', alpha=0.7)
            ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='Exit Threshold')
            ax1.axvline(x=-0.5, color='green', linestyle='--', alpha=0.7)
            ax1.legend()
        
        # 2. Signal frequency
        ax2 = axes[0, 1]
        signal_counts = {'Long': 0, 'Short': 0, 'No Position': 0}
        
        for result in signal_results:
            signals = result['signals']
            signal_counts['Long'] += (signals == 1).sum()
            signal_counts['Short'] += (signals == -1).sum()
            signal_counts['No Position'] += (signals == 0).sum()
        
        if sum(signal_counts.values()) > 0:
            labels = list(signal_counts.keys())
            values = list(signal_counts.values())
            colors = ['green', 'red', 'gray']
            
            ax2.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Signal Distribution')
        
        # 3. Spread mean reversion
        ax3 = axes[1, 0]
        if signal_results:
            # Use first pair for illustration
            result = signal_results[0]
            spread = result['spread']
            z_score = result['z_score']
            
            # Plot spread and z-score
            ax3_twin = ax3.twinx()
            
            line1 = ax3.plot(spread.index, spread.values, 'b-', alpha=0.7, label='Spread')
            line2 = ax3_twin.plot(z_score.index, z_score.values, 'r-', alpha=0.7, label='Z-Score')
            
            ax3.set_title(f'Spread and Z-Score: {result["ticker1"]}-{result["ticker2"]}')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Spread', color='b')
            ax3_twin.set_ylabel('Z-Score', color='r')
            
            # Add threshold lines
            ax3_twin.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)
            ax3_twin.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)
            ax3_twin.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
            ax3_twin.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left')
        
        # 4. Correlation vs Performance
        ax4 = axes[1, 1]
        correlations = []
        returns = []
        
        for result in signal_results:
            correlations.append(result['correlation'])
            # Calculate simple return for this pair
            strategy_returns = result['strategy_returns'].dropna()
            if len(strategy_returns) > 0:
                total_return = (1 + strategy_returns).prod() - 1
                returns.append(total_return)
            else:
                returns.append(0)
        
        if correlations and returns:
            ax4.scatter(correlations, returns, s=100, alpha=0.7)
            ax4.set_title('Correlation vs Total Return')
            ax4.set_xlabel('Correlation')
            ax4.set_ylabel('Total Return')
            ax4.grid(True, alpha=0.3)
            
            # Add pair labels
            for i, result in enumerate(signal_results):
                pair_name = f"{result['ticker1']}-{result['ticker2']}"
                ax4.annotate(pair_name, (correlations[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Config.get_results_path('phase4_signal_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating signal analysis plots: {str(e)}")

def generate_summary_and_recommendations(report_summary, signal_results):
    """Generate summary and recommendations"""
    
    logger = get_logger('Phase4', 'phase4.log')
    
    portfolio_metrics = report_summary['portfolio_metrics']
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY AND RECOMMENDATIONS")
    logger.info("=" * 60)
    
    # Overall assessment
    sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
    max_drawdown = portfolio_metrics.get('max_drawdown', 0)
    win_rate = portfolio_metrics.get('win_rate', 0)
    
    logger.info("\nOverall Strategy Assessment:")
    
    if sharpe_ratio > 1.0:
        logger.info("✓ Excellent risk-adjusted returns (Sharpe > 1.0)")
    elif sharpe_ratio > 0.5:
        logger.info("✓ Good risk-adjusted returns (Sharpe > 0.5)")
    elif sharpe_ratio > 0:
        logger.info("⚠ Positive but modest risk-adjusted returns")
    else:
        logger.info("✗ Negative risk-adjusted returns")
    
    if abs(max_drawdown) < 0.1:
        logger.info("✓ Low maximum drawdown (< 10%)")
    elif abs(max_drawdown) < 0.2:
        logger.info("⚠ Moderate maximum drawdown (10-20%)")
    else:
        logger.info("✗ High maximum drawdown (> 20%)")
    
    if win_rate > 0.6:
        logger.info("✓ High win rate (> 60%)")
    elif win_rate > 0.5:
        logger.info("⚠ Moderate win rate (50-60%)")
    else:
        logger.info("✗ Low win rate (< 50%)")
    
    # Pair-specific recommendations
    logger.info("\nPair-Specific Recommendations:")
    
    for pair_summary in report_summary['pair_summaries']:
        pair_name = pair_summary['pair']
        sharpe = pair_summary['sharpe_ratio']
        max_dd = pair_summary['max_drawdown']
        trades = pair_summary['total_trades']
        
        logger.info(f"\n{pair_name}:")
        
        if sharpe > 0.5 and abs(max_dd) < 0.15 and trades > 10:
            logger.info("  ✓ Strong candidate for live trading")
        elif sharpe > 0 and trades > 5:
            logger.info("  ⚠ Consider with caution - monitor closely")
        else:
            logger.info("  ✗ Not recommended for live trading")
        
        logger.info(f"    Sharpe: {sharpe:.3f}, MaxDD: {max_dd:.2%}, Trades: {trades}")
    
    # Parameter optimization suggestions
    logger.info("\nParameter Optimization Suggestions:")
    logger.info("1. Consider adjusting entry/exit thresholds based on volatility")
    logger.info("2. Test different rolling window sizes for z-score calculation")
    logger.info("3. Implement dynamic position sizing based on volatility")
    logger.info("4. Add additional filters (e.g., volume, market regime)")
    
    # Risk management recommendations
    logger.info("\nRisk Management Recommendations:")
    logger.info("1. Implement position limits per pair")
    logger.info("2. Add correlation-based position sizing")
    logger.info("3. Consider market regime filters")
    logger.info("4. Implement stop-loss mechanisms")
    logger.info("5. Monitor pair stability over time")
    
    # Next steps
    logger.info("\nNext Steps:")
    logger.info("1. Phase 5: Live Trading Implementation (if results are satisfactory)")
    logger.info("2. Parameter optimization using walk-forward analysis")
    logger.info("3. Additional pair selection criteria")
    logger.info("4. Risk management framework implementation")
    logger.info("5. Performance attribution analysis")

if __name__ == "__main__":
    main() 