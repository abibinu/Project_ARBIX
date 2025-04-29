"""
Main execution script for paper trading
"""
import time
import signal
import sys
from datetime import datetime, timedelta
import pandas as pd

import live_config as config
from paper_trader import PaperTrader

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

    # Initialize paper trader
    paper_trader = PaperTrader()
    
    # Load previous state if exists
    paper_trader.load_state()
    
    print(f"\nStarting paper trading for {config.SYMBOL}")
    print(f"Initial balance: ${paper_trader.balance:.2f}")
    print(f"Update interval: {config.UPDATE_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            # Update market data and check positions
            current_price = paper_trader.update_market_data()
            
            if current_price:
                # Print basic status
                print(f"\r{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"Price: ${current_price:.4f} | "
                      f"Balance: ${paper_trader.balance:.2f} | "
                      f"Active Positions: {len(paper_trader.positions)}", end="")
                
                # Save state periodically (every hour)
                if (not paper_trader.last_update or 
                    datetime.now() - paper_trader.last_update > timedelta(hours=1)):
                    paper_trader.save_state()
            
            # Sleep until next update
            time.sleep(config.UPDATE_INTERVAL)
            
        except Exception as e:
            print(f"\nError in main loop: {e}")
            time.sleep(config.UPDATE_INTERVAL)  # Wait before retrying