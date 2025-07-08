# Pairs Trading Model

A robust pairs trading model with comprehensive backtesting, cointegration testing, and risk management capabilities.

## Project Overview

This project implements a sophisticated pairs trading strategy that:

- **Correlation-based pair selection**: Analyzes all S&P 500 stocks to find the most correlated pairs
- **Cointegration testing**: Ensures selected pairs have stable long-term relationships
- **Risk management**: Comprehensive risk controls and position sizing
- **Cost optimization**: Controls trade frequency based on transaction costs
- **Daily frequency**: Uses daily data for robust analysis
- **Backtesting framework**: Validates strategy performance across different market conditions

## Project Structure

```
Pairs Trading/
├── config.py                 # Configuration parameters
├── requirements.txt          # Python dependencies
├── phase1_main.py           # Phase 1 main script
├── README.md                # This file
├── data/                    # Data processing modules
│   ├── sp500_universe.py    # S&P 500 universe management
│   └── data_collector.py    # Historical data collection
├── utils/                   # Utility modules
│   └── logger.py           # Logging functionality
├── backtesting/            # Backtesting framework
│   └── simple_backtest.py  # Phase 1 backtesting
├── data/                   # Data storage
├── logs/                   # Log files
└── results/                # Results and outputs
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Pairs Trading"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**:
   ```bash
   python -c "from config import Config; Config.create_directories()"
   ```

## Usage

### Phase 1: Data Infrastructure & Collection

Run the complete Phase 1 pipeline:

```bash
python phase1_main.py
```

Or run a quick test with a small subset of stocks:

```bash
python phase1_main.py --quick-test
```

### Configuration

Modify parameters in `config.py`:

- **Data Collection**: Start/end dates, frequency, quality thresholds
- **S&P 500 Universe**: Market cap and volume filters
- **Backtesting**: Test periods and parameters
- **Risk Management**: Position limits and thresholds

## Phase 1 Components

### 1. Environment Setup
- Project structure creation
- Configuration management
- Logging system setup

### 2. S&P 500 Universe Creation
- Fetches current S&P 500 constituents from Wikipedia
- Collects metadata (market cap, volume, sector) using yfinance
- Applies filters for minimum market cap and volume
- Saves filtered universe for downstream processing

### 3. Historical Data Pipeline
- Fetches 3+ years of daily OHLCV data
- Cleans data (removes missing values, zero prices/volumes)
- Calculates data quality scores
- Filters by quality thresholds
- Saves processed data to files

### 4. Backtesting Framework
- Tests data quality, completeness, and consistency
- Generates comprehensive reports
- Validates data suitability for Phase 2

## Key Features

### Clean Code Structure
- **Modular design**: Each component is a separate class
- **Comprehensive comments**: All functions and classes documented
- **Error handling**: Robust exception handling throughout
- **Logging**: Detailed logging for debugging and monitoring

### Configurable Parameters
- **Easy modification**: All parameters in `config.py`
- **Flexible data sources**: Can easily switch between APIs
- **Adjustable timeframes**: Configurable start/end dates and frequency
- **Quality thresholds**: Adjustable data quality requirements

### Fail-safe Mechanisms
- **Retry logic**: Automatic retries for API calls
- **Data validation**: Multiple quality checks
- **Graceful degradation**: Continues processing even if some stocks fail
- **Comprehensive logging**: Tracks all operations and errors

### Backtesting Integration
- **Real-time validation**: Tests after each major step
- **Quality metrics**: Data quality, completeness, consistency
- **Performance tracking**: Monitors processing efficiency
- **Result storage**: Saves all test results for analysis

## Output Files

### Data Files
- `data/sp500_filtered_universe.csv`: Filtered S&P 500 constituents
- `data/phase1_price_data_*.csv`: Individual stock price data
- `data/phase1_price_data_combined.csv`: Combined price data
- `data/phase1_data_summary.csv`: Data summary statistics

### Log Files
- `logs/phase1_main.log`: Main execution log
- `logs/sp500_universe.log`: Universe creation log
- `logs/data_collector.log`: Data collection log
- `logs/simple_backtest.log`: Backtesting log

### Results Files
- `results/phase1_backtest_results.json`: Comprehensive backtest results
- `results/quick_test_results.json`: Quick test results (if applicable)

## Next Steps

Phase 1 provides the foundation for:

1. **Phase 2**: Correlation Analysis & Pair Selection
2. **Phase 3**: Cointegration Analysis
3. **Phase 4**: Trading Signal Generation
4. **Phase 5**: Advanced Backtesting
5. **Phase 6**: Risk Management & Monitoring

## Troubleshooting

### Common Issues

1. **API Rate Limits**: The code includes rate limiting, but if you encounter issues, increase the sleep time in `data_collector.py`

2. **Missing Data**: Some stocks may have insufficient data - the quality filtering will handle this automatically

3. **Memory Issues**: For large datasets, consider processing in smaller chunks by modifying `Config.CHUNK_SIZE`

### Log Analysis

Check the log files in the `logs/` directory for detailed information about:
- Data collection progress
- Quality filtering results
- Error messages and warnings
- Performance metrics

## Contributing

1. Follow the existing code structure and style
2. Add comprehensive comments and documentation
3. Include backtesting for new components
4. Update configuration parameters as needed
5. Test thoroughly before committing

## License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations before using for actual trading. 