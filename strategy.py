# strategy.py
"""
Generates entry/exit signals:
  - Buy on EMA crossover + RSI + volatility + long-term trend filter
  - No EMA-signal exit (we’ll only use SL/TP)
"""
import pandas as pd
import config

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.COL_SIGNAL] = 0

    short_ema = df[config.COL_EMA_SHORT]
    long_ema  = df[config.COL_EMA_LONG]
    lterm_ema = df[config.COL_EMA_LONGTERM]
    rsi       = df[config.COL_RSI]
    atr       = df[config.COL_ATR]

    ema_diff  = short_ema - long_ema
    # Volatility filter: avoid extreme spikes
    atr_avg   = atr.rolling(window=config.ATR_PERIOD, min_periods=1).mean()
    vol_filt  = atr < (atr_avg * 1.5)

    # Entry: short EMA crosses above long EMA, RSI > threshold, price > long-term EMA, and vol ok
    buy_mask = (
        (ema_diff > 0) &
        (ema_diff.shift() <= 0) &
        (df['close'] > lterm_ema) &
        (rsi > config.RSI_BUY_THRESHOLD) &
        vol_filt
    )

    df.loc[buy_mask, config.COL_SIGNAL] = 1

    # We do NOT generate a -1 here; exits are only via SL/TP in backtester
    return df
