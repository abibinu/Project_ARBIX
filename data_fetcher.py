# data_fetcher.py
"""
Handles Binance API connection and data fetching/processing.
"""
import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import time # Import time for potential sleep
import datetime # Import datetime
import config # Import our configuration

load_dotenv()

# --- Milliseconds Conversion Helper ---
# Dictionary mapping interval strings to milliseconds
interval_to_milliseconds_map = {
    '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000,
    '1h': 3600000, '2h': 7200000, '4h': 14400000, '6h': 21600000, '8h': 28800000,
    '12h': 43200000, '1d': 86400000, '3d': 259200000, '1w': 604800000,
    # '1M' is tricky due to varying month lengths, use with caution or specific handling
}

def interval_to_milliseconds(interval_str):
    """Converts interval string to milliseconds, returns None if invalid."""
    return interval_to_milliseconds_map.get(interval_str)
# --- End Helper ---


def get_binance_client():
    """Initializes and returns the Binance client based on config."""
    # Get API keys based on mode
    if config.USE_TESTNET:
        api_key = os.getenv(config.TESTNET_API_KEY_ENV)
        api_secret = os.getenv(config.TESTNET_API_SECRET_ENV)
        print("Connecting to Binance Testnet...")
    else:
        api_key = os.getenv(config.LIVE_API_KEY_ENV)
        api_secret = os.getenv(config.LIVE_API_SECRET_ENV)
        print("Connecting to Binance Live API...")

    if not api_key or not api_secret:
        env_key = config.TESTNET_API_KEY_ENV if config.USE_TESTNET else config.LIVE_API_KEY_ENV
        print(f"Error: Binance API keys not found in environment ({env_key})")
        return None

    try:
        # For data fetching, we might want to use live API even in testnet mode
        if config.USE_TESTNET_FOR_DATA:
            client = Client(api_key, api_secret, testnet=config.USE_TESTNET)
        else:
            # Use live API for data but keep testnet flag for trading
            live_key = os.getenv(config.LIVE_API_KEY_ENV)
            live_secret = os.getenv(config.LIVE_API_SECRET_ENV)
            if live_key and live_secret:
                client = Client(live_key, live_secret, testnet=False)
            else:
                client = Client(api_key, api_secret, testnet=config.USE_TESTNET)

        # Test connection and API key validity
        try:
            # Try to get account info as this requires valid API keys
            client.get_account()
            api_type = "Testnet" if config.USE_TESTNET else "Live API"
            print(f"Successfully connected to Binance {api_type}")
            return client
        except Exception as api_error:
            print(f"Error validating API keys: {api_error}")
            return None

    except Exception as e:
        api_type = "Testnet" if config.USE_TESTNET else "Live API"
        print(f"Error connecting to Binance {api_type}: {e}")
        return None

def process_klines_to_df(klines):
    """Processes raw kline list into a formatted DataFrame."""
    if not klines:
        return pd.DataFrame()
    # Define standard columns expected by the rest of the code
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=columns)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df.set_index('timestamp', inplace=True)
    return df

# <<< NEW Function for Chunked Fetching >>>
def fetch_historical_data_chunked(client, symbol, interval, start_str, end_str=None):
    """
    Fetches historical klines in chunks between start_str and end_str.
    Uses get_historical_klines for date range fetching.
    """
    if client is None:
        print("Error: Binance client not initialized.")
        return None

    print(f"\nFetching historical data for {symbol} ({interval}) from {start_str} to {end_str or 'now'}...")

    # Convert interval to Binance constant string if needed (or use string directly)
    # Using string directly is fine for get_historical_klines
    binance_interval_str = interval

    # Max limit per request (Binance standard)
    limit_per_req = 1000
    all_klines_list = []
    fetch_count = 0

    # Use start_str directly as get_historical_klines handles string dates
    current_start_str = start_str

    while True:
        fetch_count += 1
        print(f"Fetching chunk {fetch_count} starting from {current_start_str}...")
        try:
            # Fetch data for the current chunk
            klines_chunk = client.get_historical_klines(
                symbol,
                binance_interval_str,
                current_start_str,
                end_str=end_str, # Pass the overall end date if specified
                limit=limit_per_req
            )

            if not klines_chunk:
                print("No more data found for the period.")
                break # Exit loop if no data returned

            all_klines_list.extend(klines_chunk)

            # Determine the start time for the next chunk
            last_kline_close_time_ms = klines_chunk[-1][6] # Index 6 is close_time
            next_start_time_ms = last_kline_close_time_ms + 1 # Start next fetch right after

            # Convert ms back to string format for the next API call
            next_start_dt = pd.to_datetime(next_start_time_ms, unit='ms')
            current_start_str = next_start_dt.strftime("%d %b %Y %H:%M:%S") # Format API understands

            # Optional: Check if we've gone past the end_str if one was provided
            if end_str:
                end_dt = pd.to_datetime(end_str)
                if next_start_dt >= end_dt:
                    print("Reached specified end date.")
                    break

            # Optional: Add a small delay to avoid hitting API rate limits aggressively
            time.sleep(0.3) # Sleep for 300ms between requests

        except BinanceAPIException as e:
            print(f"Binance API Exception during fetch: {e}")
            # Consider more robust error handling (e.g., retries, longer sleep)
            time.sleep(5) # Wait longer after an API error
            continue # Try fetching the same chunk again (or implement break logic)
        except Exception as e:
            print(f"An unexpected error occurred during data fetching: {e}")
            import traceback
            traceback.print_exc()
            return None # Non-recoverable error

    if not all_klines_list:
        print("Error: No data fetched despite successful connection.")
        return pd.DataFrame()

    # Process the combined list of klines
    print(f"\nTotal raw klines fetched: {len(all_klines_list)}")
    df_combined = process_klines_to_df(all_klines_list)

    # Remove potential duplicates resulting from overlapping fetches (if any)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    print(f"DataFrame shape after processing and deduplication: {df_combined.shape}")

    return df_combined
# <<< END NEW Function >>>


# (Keep the old fetch_klines function for now, maybe rename it or remove later if unused)
def fetch_klines(client, symbol, interval, limit):
    """ (Original function - kept for reference or potential use) """
    # ... (original implementation) ...
    pass # Replace 'pass' with the original implementation if you want to keep it accessible