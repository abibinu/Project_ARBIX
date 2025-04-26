# strategy.py
"""
Defines the strategy logic for generating trade signals.
"""
import pandas as pd
import config # Import config for parameters and column names

def generate_signals(df):
    """Generates Buy (1) and Sell (-1) signals based on strategy rules."""
    if df.empty:
        print("Warning: Cannot generate signals from empty DataFrame.")
        return df

    print("\nGenerating Buy/Sell signals based on EMA, RSI...")
    df[config.COL_SIGNAL] = 0 # Initialize Signal column

    try:
        # --- Buy Signal Conditions ---
        buy_cond1 = df[config.COL_EMA_SHORT] > df[config.COL_EMA_LONG]
        # Check previous bar for crossover to avoid triggering on stable state
        buy_cond2 = df[config.COL_EMA_SHORT].shift(1) <= df[config.COL_EMA_LONG].shift(1)
        buy_cond3 = df[config.COL_RSI] > config.RSI_BUY_THRESHOLD
        buy_cond4 = df[config.COL_RSI] < config.RSI_OVERBOUGHT
        buy_signal = buy_cond1 & buy_cond2 & buy_cond3 & buy_cond4
        df.loc[buy_signal, config.COL_SIGNAL] = 1

        # --- Sell Signal Conditions (EMA Crossover Exit) ---
        sell_cond1 = df[config.COL_EMA_SHORT] < df[config.COL_EMA_LONG]
        # Check previous bar for crossover
        sell_cond2 = df[config.COL_EMA_SHORT].shift(1) >= df[config.COL_EMA_LONG].shift(1)
        sell_signal = sell_cond1 & sell_cond2
        df.loc[sell_signal, config.COL_SIGNAL] = -1

        print("Signals generated.")
        return df

    except KeyError as e:
        print(f"Error generating signals: Missing expected column - {e}")
        print("Check if indicators were calculated correctly.")
        return df # Return original df if error
    except Exception as e:
        print(f"Error generating signals: {e}")
        import traceback
        traceback.print_exc()
        return df