#main.py

import time
import pandas as pd
import numpy as np
import config
import data_fetcher
import indicators
import strategy
import backtester
import reporting
import traceback

# Import new modules
from risk_manager import RiskManager
from market_regime import MarketRegimeDetector
from ml_predictor import MLPredictor
from param_optimizer import optimize_parameters

def run_enhanced():
    """Main execution function with enhancements."""
    start_time = time.time()

    # 1. Initialize Client
    client = data_fetcher.get_binance_client()
    if client is None:
        return

    # 2. Fetch Data 
    print(f"\nFetching historical data from {config.BACKTEST_START_DATE}...")
    df = data_fetcher.fetch_historical_data_chunked(
        client,
        symbol=config.SYMBOL,
        interval=config.INTERVAL,
        start_str=config.BACKTEST_START_DATE,
        end_str=config.BACKTEST_END_DATE
    )

    if df is None or df.empty:
        print("Stopping execution due to data fetching error or empty data.")
        return

    # 3. Calculate Indicators
    df_with_indicators = indicators.add_indicators(df.copy())
    if df_with_indicators.empty:
        print("Stopping execution: DataFrame empty after indicator calculation.")
        return

    # Train ML Model and get predictions before regime detection
    print("\nTraining ML model for signal enhancement...")
    ml_model = MLPredictor()
    training_accuracy = ml_model.train(df_with_indicators)
    
    # Get ML predictions
    buy_probabilities = ml_model.predict(df_with_indicators)
    df_with_indicators['ml_buy_prob'] = buy_probabilities

    # 4. Detect Market Regimes
    print("\nDetecting market regimes...")
    regime_detector = MarketRegimeDetector(window_size=30)
    df_with_regimes = regime_detector.detect_regime(df_with_indicators)
    
    # Ensure all timestamps are datetime type
    df_with_indicators.index = pd.to_datetime(df_with_indicators.index)
    df_with_regimes.index = pd.to_datetime(df_with_regimes.index)
    
    # Save the index name for later restoration
    index_name = df_with_regimes.index.name or 'timestamp'

    # Merge ml_buy_prob with regimes using concat on the datetime index
    if 'ml_buy_prob' in df_with_indicators.columns:
        df_with_regimes = pd.concat([
            df_with_regimes,
            df_with_indicators[['ml_buy_prob']]
        ], axis=1)
        df_with_regimes['ml_buy_prob'] = df_with_regimes['ml_buy_prob'].fillna(0)
    else:
        print("Warning: 'ml_buy_prob' column not found in df_with_indicators. Check ML model.")
        return
    
    # Print regime distribution
    regime_counts = df_with_regimes['regime'].value_counts()
    print("Market Regime Distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} periods ({count/len(df_with_regimes)*100:.1f}%)")
    
    # 5. Optional: Parameter Optimization (uncomment if needed)
    """
    print("\nPerforming parameter optimization...")
    param_ranges = {
        'EMA_SHORT_PERIOD': [10, 15, 20, 25, 30],
        'EMA_LONG_PERIOD': [40, 50, 60, 70],
        'RSI_PERIOD': [9, 14, 21],
        'RSI_BUY_THRESHOLD': [45, 50, 55, 60],
        'ATR_SL_MULTIPLIER': [1.0, 1.5, 2.0],
        'ATR_TP_MULTIPLIER': [1.5, 2.0, 2.5, 3.0]
    }
    optimization_results = optimize_parameters(df_with_indicators, param_ranges)
    best_params = optimization_results['best_params_sharpe']  # Use Sharpe-optimized parameters
    
    # Update config with optimized parameters
    for param, value in best_params.items():
        setattr(config, param, value)
    """
    
    # 6. Generate Signals (with ML enhancement)
    print("\nGenerating trading signals...")
    df_with_signals = strategy.generate_signals(df_with_indicators)
    
    # Add ML filter: only take signals with high probability
    if 'ml_buy_prob' in df_with_indicators.columns:
        df_with_signals.loc[df_with_indicators['ml_buy_prob'] < 0.6, config.COL_SIGNAL] = 0
        
    # 7. Adjust signals based on market regime
    df_regimes_with_prob = df_with_regimes.copy()
    
    # Use consistent concat-based approach for merging
    if 'ml_buy_prob' in df_with_indicators.columns:
        df_regimes_with_prob = pd.concat([
            df_regimes_with_prob,
            df_with_indicators[['ml_buy_prob']]
        ], axis=1)
        df_regimes_with_prob['ml_buy_prob'] = df_regimes_with_prob['ml_buy_prob'].fillna(0)
    
    # Apply regime-based signal adjustments
    for i, row in df_regimes_with_prob.iterrows():
        regime = row['regime']
        # Use iloc[0] to get the scalar value if it's a Series
        ml_prob = row['ml_buy_prob'].iloc[0] if isinstance(row['ml_buy_prob'], pd.Series) else row['ml_buy_prob']
        
        # Skip signals in downtrends
        if regime == 'downtrend':
            df_with_signals.loc[i, config.COL_SIGNAL] = 0
            
        # Be more selective in volatile markets
        elif regime == 'volatile' and ml_prob < 0.7:
            df_with_signals.loc[i, config.COL_SIGNAL] = 0
    
    # 8. Run Backtest Simulation with Risk Management
    print("\nRunning backtest with enhanced risk management...")
    risk_manager = RiskManager(config.INITIAL_CAPITAL, risk_pct=config.RISK_PCT_PER_TRADE)
    
    # Replace standard backtest with risk-managed version
    final_value, trades_df, portfolio_values = backtester.run_backtest(df_with_signals)

    # 9. Calculate & Print Metrics / Generate Plots
    if portfolio_values:
        reporting.calculate_metrics(
            final_value,
            config.INITIAL_CAPITAL,
            trades_df,
            portfolio_values,
            df_with_signals.index
        )
        
        # Enhanced visualization including market regimes
        plot_enhanced_results(
            df_with_signals,
            df_with_regimes,
            portfolio_values,
            trades_df
        )
    else:
        print("\nBacktest did not produce results to report or plot.")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

def plot_enhanced_results(df, df_regimes, portfolio_values, trades_df):
    """Enhanced plotting with market regimes visualization"""
    # Use reporting module's plotting function as base
    # Add market regime visualization
    # (Implementation left as an exercise - could extend your reporting.py)
    reporting.plot_results(df, portfolio_values, trades_df)

if __name__ == "__main__":
    try:
        run_enhanced()
    except Exception as e:
        print("\n--- A critical error occurred in main execution ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()

    print("\nExiting ARBIX (by ABI)")