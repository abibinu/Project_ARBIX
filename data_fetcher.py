# data_fetcher.py
"""
Handles Binance API connection and data fetching/processing.
"""
import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import config # Import our configuration

load_dotenv() # Load .env variables

def get_binance_client():
    """Initializes and returns the Binance client based on config."""
    use_testnet = config.USE_TESTNET_FOR_DATA
    if use_testnet:
        # Logic for Testnet keys if you add them back to config.py
        # api_key = os.getenv(config.TESTNET_API_KEY_ENV)
        # api_secret = os.getenv(config.TESTNET_API_SECRET_ENV)
        # print("Attempting to connect to Binance Testnet...")
        # if not api_key or not api_secret:
        #     print("Error: Binance Testnet API keys not found.")
        #     return None
        # Placeholder - add Testnet key names to config if needed
        print("Error: Testnet connection not fully configured in config.py")
        return None

    else: # Use Live API (READ-ONLY assumed for fetching data)
        api_key = os.getenv(config.LIVE_API_KEY_ENV)
        api_secret = os.getenv(config.LIVE_API_SECRET_ENV)
        print("Attempting to connect to Binance LIVE API (using configured keys)...")
        if not api_key or not api_secret:
            print(f"Error: Binance LIVE API keys ({config.LIVE_API_KEY_ENV}, {config.LIVE_API_SECRET_ENV}) not found.")
            return None

    try:
        client = Client(api_key, api_secret, testnet=use_testnet)
        client.ping()
        api_type = "Testnet" if use_testnet else "LIVE API"
        print(f"Successfully connected to Binance {api_type}.")
        return client
    except Exception as e:
        api_type = "Testnet" if use_testnet else "LIVE API"
        print(f"Error connecting to Binance {api_type}: {e}")
        return None

def fetch_klines(client, symbol, interval, limit):
    """Fetches klines and processes them into a DataFrame."""
    if client is None:
        print("Error: Binance client not initialized.")
        return None
    try:
        print(f"\nFetching {limit} candlesticks for {symbol} ({interval})...")
        # Use constant from Client class if needed, or pass string directly
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE, '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE, '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE, '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
            '6h': Client.KLINE_INTERVAL_6HOUR, '8h': Client.KLINE_INTERVAL_8HOUR,
            '12h': Client.KLINE_INTERVAL_12HOUR, '1d': Client.KLINE_INTERVAL_1DAY,
            '3d': Client.KLINE_INTERVAL_3DAY, '1w': Client.KLINE_INTERVAL_1WEEK,
            '1M': Client.KLINE_INTERVAL_1MONTH
        }
        binance_interval = interval_map.get(interval, interval) # Use mapped value or original string

        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=limit)
        print(f"Successfully fetched {len(klines)} candlesticks.")

        if not klines:
            print("Warning: No kline data fetched.")
            return pd.DataFrame() # Return empty DataFrame

        # Process into DataFrame
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

    except Exception as e:
        print(f"Error fetching klines for {symbol} ({interval}): {e}")
        return None

# --- Add function for chunked data fetching later ---
# def fetch_historical_data(client, symbol, interval, start_str, end_str=None):
#     pass