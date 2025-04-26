# config.py
"""
Central configuration file for Arbix Backtester
"""

# --- API Configuration ---
# Names of environment variables storing the API keys
# Use LIVE keys for fetching longer history for backtesting
LIVE_API_KEY_ENV = "BINANCE_LIVE_API_KEY"
LIVE_API_SECRET_ENV = "BINANCE_LIVE_API_SECRET"
# Set to True to use Testnet, False for Live API (for data fetching)
USE_TESTNET_FOR_DATA = False # <<< Set to False to use Live API keys

# --- Data Fetching Parameters ---
SYMBOL = 'BTCUSDT'
INTERVAL = '4h' # Use string format recognized by python-binance Client
# Limit for single API call (Binance default is 500, max 1000)
# We'll handle fetching more data in chunks later if needed.
FETCH_LIMIT = 1000

# --- Strategy Parameters ---
# Indicator Settings
EMA_SHORT_PERIOD = 12
EMA_LONG_PERIOD = 26
RSI_PERIOD = 14
ATR_PERIOD = 14

# Strategy Thresholds/Conditions
RSI_BUY_THRESHOLD = 55
RSI_SELL_THRESHOLD = 45 # Currently only used for potential sell signal modification (not active)
RSI_OVERBOUGHT = 75

# Exit Conditions
ATR_SL_MULTIPLIER = 2.5
ATR_TP_MULTIPLIER = 2.5 # Set back to 1:1 for now, can be tuned

# --- Backtesting Parameters ---
INITIAL_CAPITAL = 10000.0
TRADE_AMOUNT_USD = 1000.0 # Fixed USD amount per trade
FEE_PERCENT = 0.001 # Taker fee (0.1%)

# --- Column Names (used throughout modules) ---
# These help avoid typos and make refactoring easier
COL_EMA_SHORT = f'EMA_{EMA_SHORT_PERIOD}'
COL_EMA_LONG = f'EMA_{EMA_LONG_PERIOD}'
COL_RSI = f'RSI_{RSI_PERIOD}'
COL_ATR = f'ATRr_{ATR_PERIOD}' # Default name from pandas-ta/manual calc
COL_SIGNAL = 'Signal'