# indicators.py
"""
Functions to calculate technical indicators.
"""
import pandas as pd
import config # Import config to potentially use column names or periods

# --- Custom Indicator Functions (Moved from main script) ---
def calculate_rsi(series, period):
    """Calculate RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use rolling mean with min_periods for initial calculation
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Handle potential division by zero if avg_loss is 0
    rs = avg_gain / avg_loss.replace(0, 0.000001) # Add small epsilon to avoid div by zero

    rsi = 100 - (100 / (1 + rs))
    # Clamp RSI values between 0 and 100 (can sometimes slightly exceed due to floating point)
    rsi = rsi.clip(lower=0, upper=100)
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    # Use skipna=False to avoid issues if early data has NaNs
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    # Use rolling mean with min_periods
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def add_indicators(df):
    """Calculates and adds all required indicators to the DataFrame."""
    print("\nCalculating Technical Indicators (EMA, RSI, ATR)...")
    try:
        # Calculate EMAs using pandas built-in ewm
        df[config.COL_EMA_SHORT] = df['close'].ewm(span=config.EMA_SHORT_PERIOD, adjust=False).mean()
        df[config.COL_EMA_LONG] = df['close'].ewm(span=config.EMA_LONG_PERIOD, adjust=False).mean()

        # Calculate RSI using custom function
        df[config.COL_RSI] = calculate_rsi(df['close'], config.RSI_PERIOD)

        # Calculate ATR using custom function
        df[config.COL_ATR] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)

        print("Indicators calculated.")
        return df.dropna() # Return DataFrame after dropping NaN rows

    except Exception as e:
        print(f"Error calculating indicators: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame() # Return empty on error