"""
Main execution script for paper trading with styled terminal dashboard
"""
import time
import signal
import sys
from datetime import datetime, timedelta
import pandas as pd

import live_config as config
from paper_trader import PaperTrader
from rich.live import Live
import reporting

paper_trader = None

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print('\nShutting down paper trader...')
    if paper_trader:
        paper_trader.save_state()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Initializing paper trader...")
    # Initialize paper trader
    paper_trader = PaperTrader()
    
    # Load previous state if exists
    paper_trader.load_state()
    
    paper_trader.log("Bot initialization complete.")
    paper_trader.log(f"Paper trading started. Symbol: {config.SYMBOL}, Interval: {config.INTERVAL}")
    
    current_price = None
    next_update_s = 0

    # Start Live Dashboard
    with Live(reporting.generate_dashboard_layout(paper_trader, current_price, next_update_s, mode="Paper"), refresh_per_second=1) as live:
        while True:
            try:
                # Update market data and check positions
                current_price = paper_trader.update_market_data()
                
                # Save state periodically (every hour)
                if (not paper_trader.last_update or 
                    datetime.now() - paper_trader.last_update > timedelta(hours=1)):
                    paper_trader.save_state()
                
                # Tick down the seconds dynamically
                for s in range(config.UPDATE_INTERVAL, 0, -1):
                    next_update_s = s
                    live.update(reporting.generate_dashboard_layout(paper_trader, current_price, next_update_s, mode="Paper"))
                    time.sleep(1)
            
            except Exception as e:
                if paper_trader:
                    paper_trader.log(f"Error: {e}")
                time.sleep(config.UPDATE_INTERVAL)