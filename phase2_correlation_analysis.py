"""
Phase 2: Correlation Analysis & Pair Selection
Main execution script for correlation-based pair selection
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from data.correlation_analyzer import CorrelationAnalyzer
from utils.logger import get_logger
from config import Config

def main():
    """Main execution function for Phase 2"""
    
    # Initialize logging
    logger = get_logger('Phase2', 'phase2_correlation_analysis.log')
    logger.info("=" * 60)
    logger.info("PHASE 2: CORRELATION ANALYSIS & PAIR SELECTION")
    logger.info("=" * 60)
    
    try:
        # Initialize correlation analyzer
        analyzer = CorrelationAnalyzer()
        
        # Step 1: Load price data from Phase 1
        logger.info("Step 1: Loading price data from Phase 1...")
        price_data = analyzer.load_price_data()
        
        # Step 2: Calculate returns
        logger.info("Step 2: Calculating daily returns...")
        returns_data = analyzer.calculate_returns(price_data)
        
        # Step 3: Compute correlation matrix
        logger.info("Step 3: Computing correlation matrix...")
        correlation_matrix = analyzer.compute_correlation_matrix(
            returns_data,
            method='pearson',
            min_periods=int(len(returns_data) * 0.8)  # 80% of data required
        )
        
        # Step 4: Extract all pairs
        logger.info("Step 4: Extracting all pairs...")
        all_pairs = analyzer.get_all_pairs(correlation_matrix)
        
        # Step 5: Filter pairs based on criteria
        logger.info("Step 5: Filtering pairs...")
        filtered_pairs = analyzer.filter_pairs(
            pairs=all_pairs,
            min_correlation=0.7,      # Minimum 70% correlation
            max_correlation=0.99,     # Avoid near-perfect correlation
            exclude_same_sector=False  # We don't have sector data yet
        )
        
        # Step 6: Select top pairs with diversity
        logger.info("Step 6: Selecting top pairs...")
        selected_pairs = analyzer.select_top_pairs(
            pairs=filtered_pairs,
            n_pairs=10,               # Select top 10 pairs
            diversity_threshold=0.05  # 5% correlation difference minimum
        )
        
        # Step 7: Analyze pair stability
        logger.info("Step 7: Analyzing pair stability...")
        stable_pairs = analyzer.analyze_pair_stability(
            returns_data=returns_data,
            pairs=selected_pairs,
            window_size=252  # 1-year rolling window
        )
        
        # Step 8: Generate summary report
        logger.info("Step 8: Generating summary report...")
        summary = analyzer.generate_summary_report(stable_pairs)
        
        # Step 9: Save results
        logger.info("Step 9: Saving results...")
        analyzer.save_results(stable_pairs, 'phase2_correlation_results.json')
        
        # Save summary report
        summary_path = Config.get_results_path('phase2_summary_report.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "=" * 60)
        print("PHASE 2 RESULTS: CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"Total pairs analyzed: {summary['total_pairs_analyzed']}")
        print(f"Average correlation: {summary['avg_correlation']:.3f}")
        print(f"Average absolute correlation: {summary['avg_abs_correlation']:.3f}")
        print(f"Correlation range: {summary['min_correlation']:.3f} to {summary['max_correlation']:.3f}")
        print(f"High correlation pairs (>80%): {summary['high_correlation_pairs']}")
        print(f"Positive correlations: {summary['positive_correlations']}")
        print(f"Negative correlations: {summary['negative_correlations']}")
        
        print("\nTOP 5 SELECTED PAIRS:")
        print("-" * 40)
        for i, pair in enumerate(summary['selected_pairs'], 1):
            print(f"{i}. {pair['ticker1']} - {pair['ticker2']}")
            print(f"   Correlation: {pair['correlation']:.3f}")
            print(f"   Stability Score: {pair['stability_score']:.3f}")
            print()
        
        # Save selected pairs for Phase 3
        selected_pairs_path = Config.get_data_path('phase2_selected_pairs.csv')
        selected_df = pd.DataFrame(summary['selected_pairs'])
        selected_df.to_csv(selected_pairs_path, index=False)
        logger.info(f"Saved selected pairs to {selected_pairs_path}")
        
        logger.info("Phase 2 completed successfully!")
        logger.info(f"Results saved to: {Config.get_results_path('')}")
        
        return stable_pairs, summary
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 