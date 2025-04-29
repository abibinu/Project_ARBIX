"""
Configuration settings for paper/live trading
"""
from config import *

# --- Trading Mode ---
PAPER_TRADING = False     # Live trading enabled
USE_TESTNET = False      # Use live trading

# --- Market Update Settings ---
UPDATE_INTERVAL = 60    # Check market every minute
FETCH_TIMEOUT = 30      # Timeout for API calls

# --- Account Settings ---
INITIAL_CAPITAL = 100.0  # Conservative initial capital
MAX_TRADES = 1          # Only one trade at a time for safety
MIN_TRADE_USDT = 1     # Minimum trade size in USDT
PRICE_PRECISION = 4     # Price decimal places
QTY_PRECISION = 2       # Quantity decimal places

# --- Trading Schedule ---
TRADING_ACTIVE = True     # Global switch for trading
TRADING_HOURS = {         # Trading hours in UTC
    'start': '00:00',
    'end': '23:59'
}

# --- Risk Management ---
USE_RISK_BASED_SIZING = True
RISK_PCT_PER_TRADE = 0.30  # Increased to 30% to enable minimum size trades
MAX_DRAWDOWN = 0.15       # Adjusted to 15% to accommodate higher risk per trade
DAILY_RISK = 0.30         # Adjusted daily risk to match per-trade risk

# --- Trade Management ---
TRAILING_STOP = True       
TRAIL_AFTER_PROFIT = 0.01 # Start trailing after 1% profit
TRAIL_STOP_DISTANCE = 0.02 # 2% trailing stop distance

# --- Order Settings ---
ORDER_TYPE = 'MARKET'     # Using market orders for reliable execution
LIMIT_ORDER_PCT = 0.001   # 0.1% slippage allowance
USE_OCO_ORDERS = True     # Use OCO for take profit and stop loss

# --- Notifications ---
ENABLE_NOTIFICATIONS = True
TELEGRAM_BOT_TOKEN = "8156432800:AAEIaCfB1eYFw2AkQqvSUcjLUU-EIX2IByk"
TELEGRAM_CHAT_ID = "6498749733"

# --- Data Settings ---
KLINES_LIMIT = 500      # Number of historical candles to maintain
INDICATORS_WARMUP = 200  # Required candles for indicator calculation

# --- ML Model Settings ---
ML_RETRAIN_DAYS = 7       # Retrain model every 7 days
ML_MIN_ACCURACY = 0.60    # Minimum required accuracy
ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum probability for trade entry
ML_TRAINING_WINDOW = 500  # Number of candles to use for training

# --- Error Handling ---
MAX_API_RETRIES = 3       # Maximum API call retries
API_RETRY_DELAY = 5       # Seconds to wait between retries