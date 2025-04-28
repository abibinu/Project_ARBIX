# strategy.py
"""
Generates entry/exit signals:
  - Buy on EMA crossover + RSI confirmation
  - Sell on EMA crossover in the opposite direction + RSI confirmation
"""
import pandas as pd
import config

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.COL_SIGNAL] = 0

    short_ema = df[config.COL_EMA_SHORT]
    long_ema = df[config.COL_EMA_LONG]
    rsi = df[config.COL_RSI]

    # Simplified entry condition: EMA crossover and RSI confirmation
    buy_mask = (
        (short_ema > long_ema) &
        (rsi > config.RSI_BUY_THRESHOLD)
    )

    df.loc[buy_mask, config.COL_SIGNAL] = 1

    # Simplified exit condition: EMA crossover in the opposite direction
    sell_mask = (
        (short_ema < long_ema) &
        (rsi < config.RSI_SELL_THRESHOLD)
    )

    df.loc[sell_mask, config.COL_SIGNAL] = -1

    return df

def generate_signals_with_config(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Generates signals based on a given configuration.
    """
    df = df.copy()
    df[config.COL_SIGNAL] = 0

    short_ema = df[config.COL_EMA_SHORT]
    long_ema = df[config.COL_EMA_LONG]
    rsi = df[config.COL_RSI]

    # Entry condition: EMA crossover and RSI confirmation
    buy_mask = (
        (short_ema > long_ema) &
        (rsi > config.RSI_BUY_THRESHOLD)
    )

    df.loc[buy_mask, config.COL_SIGNAL] = 1

    # Exit condition: EMA crossover in the opposite direction
    sell_mask = (
        (short_ema < long_ema) &
        (rsi < config.RSI_SELL_THRESHOLD)
    )

    df.loc[sell_mask, config.COL_SIGNAL] = -1

    return df
