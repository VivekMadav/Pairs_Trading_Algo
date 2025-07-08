"""
Phase 1 Main Script - Data Infrastructure & Collection
Runs the complete Phase 1 pipeline with backtesting
"""

import sys
import os
from datetime import datetime
from config import Config
from utils.logger import get_logger
from data.sp500_universe import SP500Universe
from data.data_collector import DataCollector
from backtesting.simple_backtest import SimpleBacktest

def main():
    """Main function to run Phase 1"""
    
    # Initialize logging
    logger = get_logger('Phase1Main', 'phase1_main.log')
    logger.info("Starting Phase 1: Data Infrastructure & Collection")
    
    try:
        # Create necessary directories
        Config.create_directories()
        logger.info("Created project directories")
        
        # Step 1.1: Environment Setup
        logger.info("Step 1.1: Environment Setup - Complete")
        
        # Step 1.2: S&P 500 Universe Creation
        logger.info("Step 1.2: S&P 500 Universe Creation")
        sp500_universe = SP500Universe()
        
        # Fetch S&P 500 constituents
        constituents = sp500_universe.fetch_sp500_constituents()
        logger.info(f"Fetched {len(constituents)} S&P 500 constituents")
        
        # Get filtered universe
        filtered_universe = sp500_universe.get_filtered_universe()
        logger.info(f"Filtered universe contains {len(filtered_universe)} stocks")
        
        # Step 1.3: Historical Data Pipeline
        logger.info("Step 1.3: Historical Data Pipeline")
        data_collector = DataCollector()
        
        # Get tickers from filtered universe
        tickers = filtered_universe['ticker'].tolist()
        logger.info(f"Fetching historical data for {len(tickers)} tickers")
        
        # Fetch historical data
        raw_price_data = data_collector.fetch_historical_data(
            tickers=tickers,
            start_date=Config.START_DATE,
            end_date=Config.END_DATE,
            frequency=Config.DATA_FREQUENCY
        )
        
        logger.info(f"Successfully fetched data for {len(raw_price_data)} tickers")
        
        # Clean data
        cleaned_price_data = data_collector.clean_data(raw_price_data)
        logger.info(f"Successfully cleaned data for {len(cleaned_price_data)} tickers")
        
        # Filter by quality
        quality_filtered_data = data_collector.filter_by_quality(cleaned_price_data)
        logger.info(f"Quality filtering complete: {len(quality_filtered_data)} tickers passed")
        
        # Save data
        data_collector.save_data(quality_filtered_data, 'phase1_price_data')
        
        # Generate data summary
        data_summary = data_collector.get_data_summary(quality_filtered_data)
        summary_path = Config.get_data_path('phase1_data_summary.csv')
        data_summary.to_csv(summary_path, index=False)
        logger.info(f"Saved data summary to {summary_path}")
        
        # Backtesting
        logger.info("Running backtesting on collected data...")
        backtest = SimpleBacktest()
        
        # Run comprehensive backtest
        backtest_results = backtest.run_comprehensive_test(quality_filtered_data)
        
        # Save backtest results
        backtest.save_results('phase1_backtest_results.json')
        
        # Final summary
        logger.info("=== PHASE 1 COMPLETE ===")
        logger.info(f"Total tickers processed: {len(tickers)}")
        logger.info(f"Tickers with quality data: {len(quality_filtered_data)}")
        logger.info(f"Data period: {Config.START_DATE} to {Config.END_DATE}")
        logger.info(f"Data frequency: {Config.DATA_FREQUENCY}")
        
        # Check if we have sufficient data for next phase
        if len(quality_filtered_data) >= 10:  # Minimum for correlation analysis
            logger.info("SUCCESS: Sufficient data collected for Phase 2")
            return True
        else:
            logger.warning("WARNING: Insufficient data for Phase 2")
            return False
            
    except Exception as e:
        logger.error(f"Error in Phase 1: {str(e)}")
        logger.error("Phase 1 failed - check logs for details")
        return False

def run_quick_test():
    """Run a quick test with a small subset of stocks"""
    
    logger = get_logger('QuickTest', 'quick_test.log')
    logger.info("Running quick test with small subset...")
    
    try:
        # Create directories
        Config.create_directories()
        
        # Use a small subset of well-known stocks for testing
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
        logger.info(f"Testing with tickers: {test_tickers}")
        
        # Initialize components
        data_collector = DataCollector()
        backtest = SimpleBacktest()
        
        # Fetch data for test tickers
        raw_data = data_collector.fetch_historical_data(
            tickers=test_tickers,
            start_date='2023-01-01',  # Shorter period for quick test
            end_date='2023-12-31',
            frequency='1d'
        )
        
        # Clean and filter data
        cleaned_data = data_collector.clean_data(raw_data)
        quality_data = data_collector.filter_by_quality(cleaned_data)
        
        # Save test data
        data_collector.save_data(quality_data, 'quick_test_data')
        
        # Run backtest
        backtest_results = backtest.run_comprehensive_test(quality_data)
        backtest.save_results('quick_test_results.json')
        
        logger.info("Quick test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        success = run_quick_test()
    else:
        success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 