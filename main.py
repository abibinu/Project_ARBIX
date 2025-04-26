# main.py
"""
Main script to run the Arbix backtester.
Orchestrates data fetching, indicator calculation, signal generation,
backtesting, and reporting.
"""
import time
import pandas as pd
import config 
import data_fetcher
import indicators
import strategy
import backtester
import reporting
import traceback

def run():
    """Main execution function."""
    start_time = time.time()

    # 1. Initialize Client
    client = data_fetcher.get_binance_client()
    if client is None:
        return 

    # 2. Fetch Data
    df = data_fetcher.fetch_klines(
        client,
        symbol=config.SYMBOL,
        interval=config.INTERVAL,
        limit=config.FETCH_LIMIT
    )
    if df is None or df.empty:
        print("Stopping execution due to data fetching error or empty data.")
        return

    # 3. Calculate Indicators
    df_with_indicators = indicators.add_indicators(df.copy()) # Pass a copy to avoid modifying original df unintentionally
    if df_with_indicators.empty:
        print("Stopping execution: DataFrame empty after indicator calculation.")
        return

    # 4. Generate Signals
    df_with_signals = strategy.generate_signals(df_with_indicators)
    # Optional: Save df with signals for inspection
    # df_with_signals.to_csv("debug_signals.csv")

    # 5. Run Backtest Simulation
    final_value, trades_df, portfolio_values = backtester.run_backtest(df_with_signals)

    # 6. Calculate & Print Metrics / Generate Plots
    if portfolio_values: 
        reporting.calculate_metrics(
            final_value,
            config.INITIAL_CAPITAL,
            trades_df,
            portfolio_values,
            df_with_signals.index 
        )
        reporting.plot_results(
            df_with_signals, 
            portfolio_values,
            trades_df
        )
    else:
        print("\nBacktest did not produce results to report or plot.")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("\n--- A critical error occurred in main execution ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()

    print("\nMain script finished.")