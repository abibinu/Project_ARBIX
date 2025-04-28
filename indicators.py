# indicators.py
"""
Functions to calculate technical indicators.
"""
import pandas as pd
import config

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean().replace(0, 1e-6)

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    return tr.rolling(window=period, min_periods=period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds:
      - Short & long EMAs
      - Long-term EMA (trend filter)
      - RSI
      - ATR
    """
    print("\nCalculating indicators...")
    df = df.copy()

    # EMAs
    df[config.COL_EMA_SHORT]    = df['close'].ewm(span=config.EMA_SHORT_PERIOD, adjust=False).mean()
    df[config.COL_EMA_LONG]     = df['close'].ewm(span=config.EMA_LONG_PERIOD,  adjust=False).mean()
    df[config.COL_EMA_LONGTERM] = df['close'].ewm(span=config.LONG_TERM_EMA_PERIOD,
                                                 adjust=False).mean()

    # RSI & ATR
    df[config.COL_RSI] = calculate_rsi(df['close'], config.RSI_PERIOD)
    df[config.COL_ATR] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)

    # Drop initial NaNs
    return df.dropna()

def add_indicators_with_config(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Calculates and adds indicators based on a given configuration.
    """
    df = df.copy()

    # EMAs
    df[config.COL_EMA_SHORT] = df['close'].ewm(span=config.EMA_SHORT_PERIOD, adjust=False).mean()
    df[config.COL_EMA_LONG] = df['close'].ewm(span=config.EMA_LONG_PERIOD, adjust=False).mean()
    df[config.COL_EMA_LONGTERM] = df['close'].ewm(span=config.LONG_TERM_EMA_PERIOD, adjust=False).mean()

    # RSI & ATR
    df[config.COL_RSI] = calculate_rsi(df['close'], config.RSI_PERIOD)
    df[config.COL_ATR] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)

    # Drop initial NaNs
    return df.dropna()
