import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt # Import for plotting

# --- Configuration ---
# Load API keys from environment variables (still needed for client)
api_key = os.getenv("BINANCE_TESTNET_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

if not api_key or not api_secret:
    print("Error: Binance API keys not found.")
    exit()

# --- Strategy Parameters ---
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR # Use client constants for intervals
# Fetch slightly more data than needed for MA calculation buffer
limit = 100 # Number of candles to fetch (adjust as needed)
short_window = 10 # Short SMA period
long_window = 30  # Long SMA period

# --- Initialize Binance Client ---
client = Client(api_key, api_secret, testnet=True)
print("Successfully connected to Binance Testnet Client.")

# --- Fetch Historical Candlestick Data ---
try:
    print(f"\nFetching {limit} candlesticks for {symbol} ({interval})...")
    # Fetch klines (candlesticks)
    # Format: [timestamp, open, high, low, close, volume, close_time, quote_asset_vol, num_trades, taker_base_vol, taker_quote_vol, ignore]
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    print(f"Successfully fetched {len(klines)} candlesticks.")

    # --- Process Data with Pandas ---
    # Select specific columns and rename for clarity
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Keep only the columns we need for now
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Convert columns to appropriate data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Convert timestamp to datetime
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col]) # Convert price/volume to numbers

    # Set timestamp as the index (very common in time series analysis)
    df.set_index('timestamp', inplace=True)

    print("\n--- Data Head (First 5 Rows) ---")
    print(df.head())
    print("-" * 30)

    # --- Calculate Simple Moving Averages (SMAs) ---
    print(f"\nCalculating {short_window}-period and {long_window}-period SMAs...")
    df[f'SMA_{short_window}'] = df['close'].rolling(window=short_window).mean()
    df[f'SMA_{long_window}'] = df['close'].rolling(window=long_window).mean()

    print("\n--- Data Tail (Last 5 Rows with SMAs) ---")
    # Note: The first few SMA values will be NaN (Not a Number) because there isn't enough preceding data
    print(df.tail())
    print("-" * 30)

    # --- Basic Plotting (Optional, but helpful) ---
    print("\nGenerating plot...")
    plt.figure(figsize=(12, 6)) # Set the figure size
    plt.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)
    plt.plot(df.index, df[f'SMA_{short_window}'], label=f'{short_window}-Hour SMA', color='orange')
    plt.plot(df.index, df[f'SMA_{long_window}'], label=f'{long_window}-Hour SMA', color='red')

    plt.title(f'{symbol} ({interval}) Close Price and SMAs')
    plt.xlabel('Timestamp')
    plt.ylabel('Price (USDT)')
    plt.legend() # Show the legend
    plt.grid(True) # Add a grid
    plt.tight_layout() # Adjust layout
    plt.show() # Display the plot

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\nData fetching and SMA calculation script finished.")