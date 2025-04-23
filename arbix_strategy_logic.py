import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy for logic

# --- Configuration ---
api_key = os.getenv("BINANCE_TESTNET_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

if not api_key or not api_secret:
    print("Error: Binance API keys not found.")
    exit()

# --- Strategy Parameters ---
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR
limit = 200 # Let's fetch a bit more data for better signal context
short_window = 10
long_window = 30

# --- Initialize Binance Client ---
client = Client(api_key, api_secret, testnet=True)
print("Successfully connected to Binance Testnet Client.")

# --- Fetch Historical Candlestick Data ---
try:
    print(f"\nFetching {limit} candlesticks for {symbol} ({interval})...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    print(f"Successfully fetched {len(klines)} candlesticks.")

    # --- Process Data with Pandas ---
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df.set_index('timestamp', inplace=True)

    # --- Calculate SMAs ---
    print(f"\nCalculating {short_window}-period and {long_window}-period SMAs...")
    df[f'SMA_{short_window}'] = df['close'].rolling(window=short_window).mean()
    df[f'SMA_{long_window}'] = df['close'].rolling(window=long_window).mean()

    # Drop rows with NaN values created by rolling function
    df.dropna(inplace=True)

    print("\n--- Data Tail (Last 5 Rows with SMAs) ---")
    print(df.tail())
    print("-" * 30)

    # --- Implement Strategy Logic ---
    print("\nGenerating Buy/Sell signals based on SMA Crossover...")

    # Create signals based on crossover
    # .shift(1) looks at the previous row's value
    df['Signal'] = 0 # Default: 0 = Hold/Neutral
    # Condition for Buy signal (SMA10 crosses above SMA30)
    buy_condition = (df[f'SMA_{short_window}'] > df[f'SMA_{long_window}']) & \
                    (df[f'SMA_{short_window}'].shift(1) <= df[f'SMA_{long_window}'].shift(1))
    df.loc[buy_condition, 'Signal'] = 1 # 1 = Buy

    # Condition for Sell signal (SMA10 crosses below SMA30)
    sell_condition = (df[f'SMA_{short_window}'] < df[f'SMA_{long_window}']) & \
                     (df[f'SMA_{short_window}'].shift(1) >= df[f'SMA_{long_window}'].shift(1))
    df.loc[sell_condition, 'Signal'] = -1 # -1 = Sell

    # --- Simulate Position ---
    # We start neutral (position = 0). Enter long (position = 1) on Buy signal. Exit long on Sell signal.
    df['Position'] = 0 # Initialize position column
    position = 0 # Variable to track current state (0=neutral, 1=long)
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1: # Buy signal
            if position == 0: # Only buy if neutral
                position = 1
        elif df['Signal'].iloc[i] == -1: # Sell signal
            if position == 1: # Only sell if long
                position = 0
        df['Position'].iloc[i] = position # Store the state *after* the signal action

    # Shift position forward because we enter/exit based on the signal of the *current* bar,
    # but the position reflects the state *after* that bar's close.
    # This helps align signals with the position held *during* the next bar.
    df['Position'] = df['Position'].shift(1).fillna(0)


    # --- Output Signals ---
    print("\n--- Signals Generated (1 = Buy, -1 = Sell) ---")
    signals = df[df['Signal'] != 0][['close', f'SMA_{short_window}', f'SMA_{long_window}', 'Signal', 'Position']]
    print(signals)
    print("-" * 30)

    # --- Plotting with Signals ---
    print("\nGenerating plot with signals...")
    plt.figure(figsize=(14, 7))

    # Plot prices and SMAs
    plt.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.6)
    plt.plot(df.index, df[f'SMA_{short_window}'], label=f'{short_window}-Hour SMA', color='orange', linewidth=1.5)
    plt.plot(df.index, df[f'SMA_{long_window}'], label=f'{long_window}-Hour SMA', color='red', linewidth=1.5)

    # Plot Buy Signals (Green Up Arrow)
    buy_signals = df[df['Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['close'], label='Buy Signal', marker='^', color='green', s=150, alpha=1, zorder=5)

    # Plot Sell Signals (Red Down Arrow)
    sell_signals = df[df['Signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['close'], label='Sell Signal', marker='v', color='red', s=150, alpha=1, zorder=5)

    plt.title(f'{symbol} ({interval}) SMA Crossover Strategy Signals')
    plt.xlabel('Timestamp')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed error traceback

print("\nStrategy logic and signal generation script finished.")