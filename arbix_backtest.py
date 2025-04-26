import os
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import traceback

load_dotenv()

# --- Configuration ---
api_key = os.getenv("BINANCE_LIVE_API_KEY")
api_secret = os.getenv("BINANCE_LIVE_API_SECRET")

if not api_key or not api_secret:
    print("Error: Binance LIVE API keys (BINANCE_LIVE_API_KEY, BINANCE_LIVE_API_SECRET) not found in .env file.")
    print("Please generate READ-ONLY keys from your main Binance account.")
    exit()

# --- Strategy & Backtesting Parameters ---
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_4HOUR
limit = 1000 

# Indicator Parameters
ema_short_period = 12
ema_long_period = 26
rsi_period = 14
atr_period = 14
rsi_buy_threshold = 55
rsi_sell_threshold = 45
rsi_overbought = 75
atr_sl_multiplier = 2.5
atr_tp_multiplier = 2.5 

initial_capital = 10000.0
trade_amount_usd = 1000.0
fee_percent = 0.001

# --- Initialize Binance Client ---
try:
    print("Attempting to connect to Binance LIVE API (using READ-ONLY keys)...")
    client = Client(api_key, api_secret, testnet=False)
    client.ping()
    print("Successfully connected to Binance LIVE API.")
except Exception as e:
    print(f"Error connecting to Binance LIVE API: {e}")
    print("Check your LIVE API keys and permissions (should be READ-ONLY).")
    exit()

# --- Custom Technical Indicator Functions ---
def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean() 
    avg_loss = loss.rolling(window=period, min_periods=period).mean() 
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    atr = tr.rolling(window=period, min_periods=period).mean() 
    return atr

# --- Fetch Historical Candlestick Data ---
try:
    print(f"\nFetching {limit} candlesticks for {symbol} ({interval}) from LIVE API...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    print(f"Successfully fetched {len(klines)} candlesticks.")

    if not klines:
        print("Error: No data fetched from LIVE API.")
        exit()

    # --- Process Data ---
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

    # --- Calculate Technical Indicators ---
    print("\nCalculating Technical Indicators (EMA, RSI, ATR)...")
    df[f'EMA_{ema_short_period}'] = df['close'].ewm(span=ema_short_period, adjust=False).mean()
    df[f'EMA_{ema_long_period}'] = df['close'].ewm(span=ema_long_period, adjust=False).mean()
    df[f'RSI_{rsi_period}'] = calculate_rsi(df['close'], rsi_period)
    df[f'ATRr_{atr_period}'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    df.dropna(inplace=True)

    if df.empty:
        print("Error: DataFrame empty after indicator calculation.")
        exit()

    print("\n--- Data Tail with Indicators (Last 5 Rows) ---")
    print(df.tail())
    print("-" * 30)

    # --- Define Strategy Signals ---
    print("\nGenerating Buy/Sell signals based on EMA, RSI...")
    df['Signal'] = 0
    ema_short_col = f'EMA_{ema_short_period}'
    ema_long_col = f'EMA_{ema_long_period}'
    rsi_col = f'RSI_{rsi_period}'
    buy_cond1 = df[ema_short_col] > df[ema_long_col]
    buy_cond2 = df[ema_short_col].shift(1) <= df[ema_long_col].shift(1)
    buy_cond3 = df[rsi_col] > rsi_buy_threshold
    buy_cond4 = df[rsi_col] < rsi_overbought
    buy_signal = buy_cond1 & buy_cond2 & buy_cond3 & buy_cond4
    df.loc[buy_signal, 'Signal'] = 1
    sell_cond1 = df[ema_short_col] < df[ema_long_col]
    sell_cond2 = df[ema_short_col].shift(1) >= df[ema_long_col].shift(1)
    sell_signal = sell_cond1 & sell_cond2
    df.loc[sell_signal, 'Signal'] = -1

    # --- Backtesting Simulation ---
    print(f"\nStarting backtesting simulation with dynamic ATR SL/TP...")
    print(f"ATR Multipliers: SL={atr_sl_multiplier}, TP={atr_tp_multiplier}")
    cash = initial_capital
    crypto_held = 0.0
    portfolio_values = []
    trades = []
    entry_price = 0.0
    stop_loss_level = 0.0
    take_profit_level = 0.0
    position_status = 0
    atr_col = f'ATRr_{atr_period}'

    for i in range(len(df)):
        current_index = df.index[i]
        current_close = df['close'].iloc[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_atr = df[atr_col].iloc[i]
        if pd.isna(current_atr) or current_atr == 0: 
             print(f"Warning {current_index}: Skipping bar due to invalid ATR ({current_atr})")
             portfolio_values.append(cash + (crypto_held * current_close)) 
             continue 

        actual_signal_this_bar = df['Signal'].iloc[i]
        current_value = cash + (crypto_held * current_close)
        portfolio_values.append(current_value)
        exit_reason = None

        if position_status == 1:
            if current_high >= take_profit_level:
                exit_price = take_profit_level
                exit_reason = 'TAKE_PROFIT'
                print(f"INFO {current_index}: TAKE-PROFIT triggered at {exit_price:.2f} (Entry: {entry_price:.2f}, High: {current_high:.2f})")
            elif current_low <= stop_loss_level:
                exit_price = stop_loss_level
                exit_reason = 'STOP_LOSS'
                print(f"INFO {current_index}: STOP-LOSS triggered at {exit_price:.2f} (Entry: {entry_price:.2f}, Low: {current_low:.2f})")
            elif actual_signal_this_bar == -1:
                exit_price = current_close
                exit_reason = 'SELL_EMA'
                print(f"INFO {current_index}: EMA SELL signal -> Selling @ {exit_price:.2f} (Entry: {entry_price:.2f})")

            if exit_reason:
                sell_value_usd = crypto_held * exit_price
                fee = sell_value_usd * fee_percent
                cash += (sell_value_usd - fee)
                sell_amount_crypto = crypto_held
                profit = (sell_value_usd - fee) - (sell_amount_crypto * entry_price)
                trades.append({'Timestamp': current_index, 'Type': exit_reason, 'Price': exit_price, 'Amount_Crypto': sell_amount_crypto, 'Received_USD': sell_value_usd - fee, 'Fee': fee, 'Profit': profit})
                crypto_held = 0.0
                entry_price = 0.0
                stop_loss_level = 0.0
                take_profit_level = 0.0
                position_status = 0

        if position_status == 0:
            if actual_signal_this_bar == 1:
                entry_price = current_close
                buy_amount_crypto = trade_amount_usd / entry_price
                fee_crypto = buy_amount_crypto * fee_percent
                buy_amount_crypto_net = buy_amount_crypto - fee_crypto
                fee_usd = fee_crypto * entry_price
                cash -= trade_amount_usd
                crypto_held += buy_amount_crypto_net

                stop_loss_level = entry_price - (atr_sl_multiplier * current_atr)
                take_profit_level = entry_price + (atr_tp_multiplier * current_atr)

                position_status = 1
                trades.append({'Timestamp': current_index, 'Type': 'BUY', 'Price': entry_price, 'Amount_Crypto': buy_amount_crypto_net, 'Cost_USD': trade_amount_usd, 'Fee': fee_usd})
                print(f"INFO {current_index}: BUY signal -> Bought {buy_amount_crypto_net:.6f} @ {entry_price:.2f}")
                print(f"   SL set at {stop_loss_level:.2f}, TP set at {take_profit_level:.2f} (ATR: {current_atr:.2f})")


    # --- Post-Loop Calculations ---
    final_value = cash + (crypto_held * df['close'].iloc[-1] if crypto_held > 0 and not df.empty else cash) # Handle empty df case
    # Ensure portfolio_values has at least one element if df was processed
    if len(df) > 0 and not portfolio_values: # If loop was skipped due to ATR=0 but df existed
         portfolio_values.append(initial_capital) # Start with initial capital
    # Append final value if it wasn't the last action in the loop
    if not df.empty and len(portfolio_values) == len(df):
         portfolio_values.append(final_value)


    # --- Calculate Performance Metrics ---
    print("\n--- Backtest Results ---")
    total_return_percent = ((final_value - initial_capital) / initial_capital) * 100
    start_date = df.index[0] if not df.empty else "N/A"
    end_date = df.index[-1] if not df.empty else "N/A"
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return_percent:.2f}%")

    if trades:
        trade_df = pd.DataFrame(trades)
        buy_trades = trade_df[trade_df['Type'] == 'BUY']
        sell_trades = trade_df[trade_df['Type'].isin(['SELL_EMA', 'STOP_LOSS', 'TAKE_PROFIT'])]
        stop_loss_exits = trade_df[trade_df['Type'] == 'STOP_LOSS']
        take_profit_exits = trade_df[trade_df['Type'] == 'TAKE_PROFIT']
        ema_sell_exits = trade_df[trade_df['Type'] == 'SELL_EMA']
        print(f"Total Trades Executed (Closed): {len(sell_trades)}")
        print(f"Number of Stop-Loss Exits: {len(stop_loss_exits)}")
        print(f"Number of Take-Profit Exits: {len(take_profit_exits)}")
        print(f"Number of EMA Crossover Exits: {len(ema_sell_exits)}")

        if not sell_trades.empty:
            profitable_trades = sell_trades[sell_trades['Profit'] > 0]
            loss_trades = sell_trades[sell_trades['Profit'] <= 0]
            win_rate = (len(profitable_trades) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
            print(f"Win Rate (Based on Closed Trades): {win_rate:.2f}%")
            total_profit = profitable_trades['Profit'].sum()
            total_loss = abs(loss_trades['Profit'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
            print(f"Total Profit from Winners: ${total_profit:.2f}")
            print(f"Total Loss from Losers: ${total_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")

        # Calculate Max Drawdown
        portfolio_index = df.index.copy()
        if not portfolio_index.empty and len(portfolio_values) > len(portfolio_index):
             portfolio_index = portfolio_index.append(pd.Index([portfolio_index[-1] + pd.Timedelta(seconds=1)] * (len(portfolio_values) - len(portfolio_index)) )) # Add padding index
        elif portfolio_index.empty and portfolio_values:
             portfolio_index = pd.Index([pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(len(portfolio_values))]) # Create index if df was empty

        # Ensure Series aligns with index
        if len(portfolio_values) == len(portfolio_index) and not portfolio_index.empty:
            portfolio_series = pd.Series(portfolio_values, index=portfolio_index)
            rolling_max = portfolio_series.cummax()
            daily_drawdown = portfolio_series / rolling_max - 1.0
            max_drawdown = daily_drawdown.cummin().min() * 100
            print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        elif not df.empty: # Only warn if df wasn't empty initially
            print(f"Warning: Portfolio values length ({len(portfolio_values)}) vs index length ({len(portfolio_index)}) mismatch. Cannot calculate drawdown reliably.")
            max_drawdown = "N/A"
        else:
            max_drawdown = 0.0 

    else:
        print("No trades were executed during this period.")
        max_drawdown = 0.0

    print("-" * 30)

    # --- Plotting Equity Curve and Signals ---
    print("\nGenerating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [4, 2, 2]})

    axes[0].plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.6, linewidth=1)
    axes[0].plot(df.index, df[ema_short_col], label=f'{ema_short_period}-Period EMA', color='orange', linewidth=1.5)
    axes[0].plot(df.index, df[ema_long_col], label=f'{ema_long_period}-Period EMA', color='red', linewidth=1.5)

    if trades:
        trade_timestamps = trade_df['Timestamp'].unique()
        valid_trade_timestamps = df.index.intersection(trade_timestamps)
        filtered_trade_df = trade_df[trade_df['Timestamp'].isin(valid_trade_timestamps)]
        buy_markers = filtered_trade_df[filtered_trade_df['Type'] == 'BUY']
        sell_ema_markers = filtered_trade_df[filtered_trade_df['Type'] == 'SELL_EMA']
        stop_loss_markers = filtered_trade_df[filtered_trade_df['Type'] == 'STOP_LOSS']
        take_profit_markers = filtered_trade_df[filtered_trade_df['Type'] == 'TAKE_PROFIT']

        if not buy_markers.empty:
            axes[0].scatter(buy_markers['Timestamp'], df.loc[buy_markers['Timestamp']]['close'], label='Buy Signal', marker='^', color='lime', s=100, alpha=1, zorder=5, edgecolors='black', linewidth=0.5)
        if not sell_ema_markers.empty:
            axes[0].scatter(sell_ema_markers['Timestamp'], df.loc[sell_ema_markers['Timestamp']]['close'], label='Sell Signal (EMA)', marker='v', color='red', s=100, alpha=1, zorder=5, edgecolors='black', linewidth=0.5)
        if not stop_loss_markers.empty:
            axes[0].scatter(stop_loss_markers['Timestamp'], df.loc[stop_loss_markers['Timestamp']]['low'], label='Stop-Loss Exit', marker='x', color='magenta', s=120, alpha=1, zorder=5, linewidth=1.5)
        if not take_profit_markers.empty:
            axes[0].scatter(take_profit_markers['Timestamp'], df.loc[take_profit_markers['Timestamp']]['high'], label='Take-Profit Exit', marker='*', color='cyan', s=150, alpha=1, zorder=5, edgecolors='black', linewidth=0.5)

    axes[0].set_title(f'{symbol} ({interval}) EMA/RSI/ATR Strategy Backtest')
    axes[0].set_ylabel('Price (USDT)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    # Plot 2: RSI
    rsi_col = f'RSI_{rsi_period}' # Ensure rsi_col is defined
    axes[1].plot(df.index, df[rsi_col], label=f'RSI ({rsi_period})', color='purple', linewidth=1)
    axes[1].axhline(rsi_overbought, color='red', linestyle='--', linewidth=1, label=f'Overbought ({rsi_overbought})')
    axes[1].axhline(rsi_buy_threshold, color='blue', linestyle=':', linewidth=1, label=f'Buy Threshold ({rsi_buy_threshold})')
    axes[1].axhline(rsi_sell_threshold, color='blue', linestyle=':', linewidth=1, label=f'Sell Threshold ({rsi_sell_threshold})')
    axes[1].axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    axes[1].set_title('Relative Strength Index (RSI)')
    axes[1].set_ylabel('RSI Value')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    # Plot 3: Portfolio Value (Equity Curve)
    if 'portfolio_series' in locals() and not portfolio_series.empty:
        axes[2].plot(portfolio_series.index, portfolio_series, label='Portfolio Value', color='green')
        axes[2].set_title('Equity Curve')
        axes[2].set_ylabel('Portfolio Value (USDT)')
        axes[2].set_xlabel('Timestamp')
        axes[2].grid(True)
        axes[2].legend(loc='upper left')
    elif not df.empty: # Check if df wasn't empty but series wasn't created
        axes[2].text(0.5, 0.5, 'Equity Curve Error (Check Data/Loop)', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title('Equity Curve')
    else: # df was empty
         axes[2].text(0.5, 0.5, 'No Trades Executed', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
         axes[2].set_title('Equity Curve')


    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Backtest Results: {symbol} ({interval})', fontsize=14)
    plt.show()


except Exception as e:
    print(f"\n--- An error occurred during script execution ---")
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {e}")
    print("\n--- Traceback ---")
    traceback.print_exc()

print("\nBacktesting script finished.")