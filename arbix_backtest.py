import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
api_key = os.getenv("BINANCE_TESTNET_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

if not api_key or not api_secret:
    print("Error: Binance API keys not found.")
    exit()

# --- Strategy & Backtesting Parameters ---
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR
limit = 1000 # Fetch more data for a more meaningful backtest
short_window = 10
long_window = 30
initial_capital = 10000.0 # Starting capital in USDT for simulation
trade_amount_usd = 1000.0 # Fixed amount of USDT to use per trade
# Binance fees (example - use 0.1% taker fee) - CHECK CURRENT BINANCE FEES!
fee_percent = 0.001 # 0.1% = 0.001

# --- Initialize Binance Client ---
client = Client(api_key, api_secret, testnet=True)
print("Successfully connected to Binance Testnet Client.")

# --- Fetch Historical Candlestick Data ---
try:
    print(f"\nFetching {limit} candlesticks for {symbol} ({interval})...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    print(f"Successfully fetched {len(klines)} candlesticks.")

    # --- Process Data ---
    # (Same data processing as before: DataFrame creation, type conversion, index setting)
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
    df.dropna(inplace=True) # Remove rows with NaN SMAs

    # --- Generate Signals & Position (Same logic as before) ---
    print("\nGenerating Buy/Sell signals...")
    df['Signal'] = 0
    buy_condition = (df[f'SMA_{short_window}'] > df[f'SMA_{long_window}']) & \
                    (df[f'SMA_{short_window}'].shift(1) <= df[f'SMA_{long_window}'].shift(1))
    df.loc[buy_condition, 'Signal'] = 1
    sell_condition = (df[f'SMA_{short_window}'] < df[f'SMA_{long_window}']) & \
                     (df[f'SMA_{short_window}'].shift(1) >= df[f'SMA_{long_window}'].shift(1))
    df.loc[sell_condition, 'Signal'] = -1

    # Determine position based on signals (0 = neutral, 1 = long)
    df['Position'] = 0
    position = 0
    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        if signal == 1 and position == 0:
            position = 1
        elif signal == -1 and position == 1:
            position = 0
        df.loc[df.index[i], 'Position'] = position
    # Shift position to reflect holding state *during* the next bar
    df['Position'] = df['Position'].shift(1).fillna(0)


    # --- Backtesting Simulation ---
    print("\nStarting backtesting simulation...")
    cash = initial_capital
    crypto_held = 0.0
    portfolio_values = [] # List to store portfolio value over time
    trades = [] # List to store trade details
    entry_price = 0.0

    for i in range(len(df)):
        current_close = df['close'].iloc[i]
        current_signal = df['Signal'].iloc[i]
        current_position_before_trade = df['Position'].iloc[i] # Position held *during* this bar (determined by *previous* signal)
        actual_signal_this_bar = df['Signal'].iloc[i] # The signal generated *at the end* of this bar

        # Calculate current portfolio value before any trade on this bar
        current_value = cash + (crypto_held * current_close)
        portfolio_values.append(current_value)

        # Execute trades based on the signal generated *at the end* of the current bar
        # Assume trade happens near the close price of the *current* bar for simplicity
        if actual_signal_this_bar == 1 and current_position_before_trade == 0: # Check position *before* trade
            # --- Execute Buy ---
            buy_amount_crypto = trade_amount_usd / current_close
            fee = buy_amount_crypto * fee_percent
            buy_amount_crypto -= fee # Subtract fee from crypto received
            cash -= trade_amount_usd # Spend cash
            crypto_held += buy_amount_crypto
            entry_price = current_close # Record entry price
            trades.append({'Timestamp': df.index[i], 'Type': 'BUY', 'Price': current_close, 'Amount_Crypto': buy_amount_crypto, 'Cost_USD': trade_amount_usd, 'Fee': fee * current_close})
            # print(f"DEBUG {df.index[i]}: BUY signal -> Bought {buy_amount_crypto:.6f} @ {current_close:.2f}. Cash: {cash:.2f}")

        elif actual_signal_this_bar == -1 and current_position_before_trade == 1: # Check position *before* trade
            # --- Execute Sell ---
            sell_value_usd = crypto_held * current_close
            fee = sell_value_usd * fee_percent
            cash += (sell_value_usd - fee) # Add cash received (less fee)
            sell_amount_crypto = crypto_held
            crypto_held = 0.0
            profit = sell_value_usd - fee - (sell_amount_crypto * entry_price) # Simple profit calc for this trade
            trades.append({'Timestamp': df.index[i], 'Type': 'SELL', 'Price': current_close, 'Amount_Crypto': sell_amount_crypto, 'Received_USD': sell_value_usd - fee, 'Fee': fee, 'Profit': profit})
            entry_price = 0.0 # Reset entry price
            # print(f"DEBUG {df.index[i]}: SELL signal -> Sold {sell_amount_crypto:.6f} @ {current_close:.2f}. Cash: {cash:.2f}")


    # Add final portfolio value after loop ends
    final_value = cash + (crypto_held * df['close'].iloc[-1])
    portfolio_values.append(final_value) # Append one last value
    df['Portfolio_Value'] = pd.Series(portfolio_values[:-1], index=df.index) # Align with df index

    # --- Calculate Performance Metrics ---
    print("\n--- Backtest Results ---")
    total_return_percent = ((final_value - initial_capital) / initial_capital) * 100
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return_percent:.2f}%")

    if trades:
        trade_df = pd.DataFrame(trades)
        buy_trades = trade_df[trade_df['Type'] == 'BUY']
        sell_trades = trade_df[trade_df['Type'] == 'SELL']
        print(f"Total Trades Executed: {len(sell_trades)}") # Count closed trades

        if not sell_trades.empty:
            profitable_trades = sell_trades[sell_trades['Profit'] > 0]
            loss_trades = sell_trades[sell_trades['Profit'] <= 0]
            win_rate = (len(profitable_trades) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
            print(f"Win Rate: {win_rate:.2f}%")

            total_profit = profitable_trades['Profit'].sum()
            total_loss = abs(loss_trades['Profit'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else np.inf # Handle division by zero
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"Total Loss: ${total_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")

        # Calculate Max Drawdown
        portfolio_series = pd.Series(portfolio_values, index=df.index.append(pd.Index([df.index[-1] + pd.Timedelta(hours=1)]))) # Add dummy index for final value
        rolling_max = portfolio_series.cummax()
        daily_drawdown = portfolio_series / rolling_max - 1.0
        max_drawdown = daily_drawdown.cummin().min() * 100
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

    else:
        print("No trades were executed during this period.")

    print("-" * 30)

    # --- Plotting Equity Curve and Signals ---
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Price, SMAs, Signals
    axes[0].plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)
    axes[0].plot(df.index, df[f'SMA_{short_window}'], label=f'{short_window}-Hour SMA', color='orange', linewidth=1.5)
    axes[0].plot(df.index, df[f'SMA_{long_window}'], label=f'{long_window}-Hour SMA', color='red', linewidth=1.5)
    buy_signals = df[df['Signal'] == 1]
    axes[0].scatter(buy_signals.index, buy_signals['close'], label='Buy Signal', marker='^', color='green', s=100, alpha=1, zorder=5)
    sell_signals = df[df['Signal'] == -1]
    axes[0].scatter(sell_signals.index, sell_signals['close'], label='Sell Signal', marker='v', color='red', s=100, alpha=1, zorder=5)
    axes[0].set_title(f'{symbol} SMA Crossover Strategy Backtest')
    axes[0].set_ylabel('Price (USDT)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Portfolio Value (Equity Curve)
    axes[1].plot(portfolio_series.index, portfolio_series, label='Portfolio Value', color='purple')
    axes[1].set_title('Equity Curve')
    axes[1].set_ylabel('Portfolio Value (USDT)')
    axes[1].set_xlabel('Timestamp')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


except Exception as e:
    print(f"\nAn error occurred during backtesting: {e}")
    import traceback
    traceback.print_exc()

print("\nBacktesting script finished.")