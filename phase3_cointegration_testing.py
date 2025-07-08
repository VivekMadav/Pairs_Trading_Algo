"""
Phase 3: Cointegration Testing
Main execution script for cointegration analysis and trading parameter optimization
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from data.cointegration_tester import CointegrationTester
from utils.logger import get_logger
from config import Config

def main():
    """Main execution function for Phase 3"""
    
    # Initialize logging
    logger = get_logger('Phase3', 'phase3_cointegration_testing.log')
    logger.info("=" * 60)
    logger.info("PHASE 3: COINTEGRATION TESTING")
    logger.info("=" * 60)
    
    try:
        # Initialize cointegration tester
        tester = CointegrationTester()
        
        # Step 1: Load price data from Phase 1
        logger.info("Step 1: Loading price data from Phase 1...")
        price_data = tester.load_price_data()
        
        # Step 2: Load selected pairs from Phase 2
        logger.info("Step 2: Loading selected pairs from Phase 2...")
        pairs_data = tester.load_selected_pairs()
        
        # Step 3: Run complete cointegration analysis
        logger.info("Step 3: Running complete cointegration analysis...")
        test_results = tester.run_complete_analysis(pairs_data)
        
        # Step 4: Generate summary report
        logger.info("Step 4: Generating summary report...")
        summary = tester.generate_summary_report(test_results)
        
        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        tester.save_results(test_results, 'phase3_cointegration_results.json')
        
        # Save summary report
        summary_path = Config.get_results_path('phase3_summary_report.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "=" * 60)
        print("PHASE 3 RESULTS: COINTEGRATION TESTING")
        print("=" * 60)
        print(f"Total pairs tested: {summary['total_pairs_tested']}")
        print(f"Tradeable pairs: {summary['tradeable_pairs']}")
        print(f"Tradeable ratio: {summary['tradeable_ratio']:.1%}")
        print(f"Average half-life: {summary['avg_half_life']:.1f} days")
        print(f"Average Hurst exponent: {summary['avg_hurst_exponent']:.3f}")
        print(f"Average stability: {summary['avg_stability']:.3f}")
        
        if summary['tradeable_pairs'] > 0:
            print("\nTRADEABLE PAIRS:")
            print("-" * 60)
            for i, pair in enumerate(summary['tradeable_pairs_details'], 1):
                print(f"{i}. {pair['ticker1']} - {pair['ticker2']}")
                print(f"   Correlation: {pair['correlation']:.3f}")
                print(f"   Half-life: {pair['half_life']:.1f} days")
                print(f"   Hurst exponent: {pair['hurst_exponent']:.3f}")
                print(f"   Stability: {pair['stability']:.3f}")
                print(f"   Entry threshold: {pair['entry_threshold']:.2f}")
                print(f"   Exit threshold: {pair['exit_threshold']:.2f}")
                print()
        else:
            print("\n⚠️  NO TRADEABLE PAIRS FOUND")
            print("Consider adjusting correlation thresholds or testing additional pairs.")
        
        # Save tradeable pairs for Phase 4
        if summary['tradeable_pairs'] > 0:
            tradeable_pairs_path = Config.get_data_path('phase3_tradeable_pairs.csv')
            tradeable_df = pd.DataFrame(summary['tradeable_pairs_details'])
            tradeable_df.to_csv(tradeable_pairs_path, index=False)
            logger.info(f"Saved tradeable pairs to {tradeable_pairs_path}")
        
        # Detailed analysis for each pair
        print("\nDETAILED ANALYSIS:")
        print("-" * 60)
        for result in test_results:
            ticker1, ticker2 = result['ticker1'], result['ticker2']
            eg = result['engle_granger']
            johansen = result['johansen']
            spread_stats = result['spread_statistics']
            thresholds = result['thresholds']
            stability = result['stability_test']
            
            print(f"\n{ticker1} - {ticker2}:")
            print(f"  Engle-Granger: p-value={eg['engle_granger_pvalue']:.4f}, cointegrated={eg['is_cointegrated']}")
            print(f"  Johansen: cointegrated={johansen['johansen_is_cointegrated']}")
            print(f"  Half-life: {spread_stats['half_life']:.1f} days")
            print(f"  Hurst exponent: {spread_stats['hurst_exponent']:.3f}")
            print(f"  Stability: {stability['overall_stability']:.3f}")
            print(f"  Tradeable: {result['is_tradeable']}")
        
        logger.info("Phase 3 completed successfully!")
        logger.info(f"Results saved to: {Config.get_results_path('')}")
        
        return test_results, summary
        
    except Exception as e:
        logger.error(f"Phase 3 failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 