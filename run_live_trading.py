"""
Live trading execution script with styled terminal dashboard
"""
import time
import sys
import signal
from datetime import datetime, timedelta
import pandas as pd

import live_config as config
from paper_trader import PaperTrader
from notifications import TelegramNotifier
from rich.live import Live
import reporting

# Initialize global variables
trader = None
last_candle_time = None
historical_data = None

def calculate_next_interval(interval: str) -> int:
    """Calculate seconds until next candle close"""
    intervals = {
        '1m': 60, '3m': 180, '5m': 300, '15m': 900,
        '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
        '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400
    }
    
    seconds = intervals.get(interval, 3600)  # Default to 1h if interval not found
    now = datetime.now()
    
    if interval == '1h':
        # For hourly interval, get next hour
        next_interval = now.replace(minute=0, second=0, microsecond=0)
        if next_interval <= now:
            next_interval += timedelta(hours=1)
    else:
        # For other intervals
        next_interval = now.replace(second=0, microsecond=0)
        while next_interval <= now:
            next_interval += timedelta(seconds=seconds)
    
    return max(1, (next_interval - now).seconds)

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global trader
    print('\nShutting down live trader...')
    notifier = TelegramNotifier()
    notifier.send_message("🛑 Bot shutting down - User initiated stop")
    if trader:
        trader.save_state()
    sys.exit(0)

def should_update_data(interval: str) -> bool:
    """Determine if we need to fetch new data based on timeframe"""
    global last_candle_time
    
    if last_candle_time is None:
        return True
        
    now = datetime.now()
    interval_seconds = {
        '1m': 60, '3m': 180, '5m': 300, '15m': 900,
        '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
        '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400
    }.get(interval, 3600)
    
    time_since_last = (now - last_candle_time).total_seconds()
    return time_since_last >= interval_seconds

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Initializing trading bot...")
    # Initialize trader
    trader = PaperTrader()
    
    # Load previous state if exists
    trader.load_state()

    mode_str = "Paper" if config.PAPER_TRADING else "Live"
    trader.log("Bot initialization complete.")
    trader.log(f"{mode_str} trading started on {'Testnet' if config.USE_TESTNET else 'Live'} API.")
    trader.log(f"Trading Pair: {config.SYMBOL}, Timeframe: {config.INTERVAL}")
    
    # Send startup notification
    notifier = TelegramNotifier()
    notifier.send_message(
        f"🚀 Trading bot started\n"
        f"Mode: {mode_str}\n"
        f"Symbol: {config.SYMBOL}\n"
        f"Interval: {config.INTERVAL}\n"
        f"Initial Balance: ${trader.balance:.2f}"
    )

    last_model_train = datetime.now()
    training_interval = timedelta(days=7)  # Retrain weekly by default
    current_price = None
    seconds_to_next = 0
    
    # Start Live Dashboard
    with Live(reporting.generate_dashboard_layout(trader, current_price, seconds_to_next, mode=mode_str), refresh_per_second=1) as live:
        while True:
            try:
                current_time = datetime.now()
                
                # Check if it's time to retrain the model
                if current_time - last_model_train >= training_interval:
                    trader.log("Retraining ML model with recent data...")
                    historical_data = trader._fetch_initial_data()
                    if not historical_data.empty:
                        trader.ml_predictor.train(historical_data)
                        last_model_train = current_time
                        trader.log("ML model retrained successfully.")
                        notifier.send_message("🧠 ML model retrained successfully")
                
                # Only fetch new data if needed based on timeframe
                if should_update_data(config.INTERVAL):
                    current_price = trader.update_market_data()
                    last_candle_time = current_time
                else:
                    current_price = trader.get_current_price()  # Just get current price without full update
                
                # Save state periodically
                trader.save_state()
                
                # Calculate time until next candle update
                seconds_to_next = calculate_next_interval(config.INTERVAL)
                
                # Countdown for next update
                for s in range(seconds_to_next, 0, -1):
                    live.update(reporting.generate_dashboard_layout(trader, current_price, s, mode=mode_str))
                    time.sleep(1)
                    # If we need a candle update immediately, break early
                    if should_update_data(config.INTERVAL):
                        break
                
            except Exception as e:
                error_msg = f"Error in main loop: {e}"
                if trader:
                    trader.log(f"Error: {error_msg}")
                    trader.save_state()
                notifier.notify_error(error_msg)
                time.sleep(10)