# config.py
"""
Central configuration file for Arbix Backtester
"""
import datetime

# --- API Configuration ---
LIVE_API_KEY_ENV      = "BINANCE_LIVE_API_KEY"
LIVE_API_SECRET_ENV   = "BINANCE_LIVE_API_SECRET"
USE_TESTNET_FOR_DATA  = False    # Use Live API for long history

# --- Data Fetching Parameters ---
SYMBOL               = 'DOGEUSDT'
INTERVAL             = '4h'       # Optimal timeframe
BACKTEST_START_DATE  = "1 Jan, 2021"
BACKTEST_END_DATE    = None      # None means fetch up to now

# --- Strategy Parameters ---
EMA_SHORT_PERIOD     = 20
EMA_LONG_PERIOD      = 50
LONG_TERM_EMA_PERIOD = 200       # New long-term trend filter
RSI_PERIOD           = 14
ATR_PERIOD           = 14

RSI_BUY_THRESHOLD    = 55
RSI_SELL_THRESHOLD   = 45
RSI_OVERBOUGHT       = 75

# --- ATR-based SL/TP (optimal configuration) ---
ATR_SL_MULTIPLIER    = 1.5       # Best performing stop loss
ATR_TP_MULTIPLIER    = 2.0       # Best performing take profit

# --- Position Sizing ---
USE_RISK_BASED_SIZING = True     
RISK_PCT_PER_TRADE    = 0.02     # Optimal risk per trade (2%)

# --- Backtesting Parameters ---
INITIAL_CAPITAL      = 10000.0   # Starting USD balance

# --- Fee & Symbol ---
FEE_PERCENT          = 0.001     # 0.1% per trade

# --- Column Names (auto-derived) ---
COL_EMA_SHORT        = f'EMA_{EMA_SHORT_PERIOD}'
COL_EMA_LONG         = f'EMA_{EMA_LONG_PERIOD}'
COL_EMA_LONGTERM     = f'EMA_{LONG_TERM_EMA_PERIOD}'
COL_RSI              = f'RSI_{RSI_PERIOD}'
COL_ATR              = f'ATRr_{ATR_PERIOD}'
COL_SIGNAL           = 'Signal'
