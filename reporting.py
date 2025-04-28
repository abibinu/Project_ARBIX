# reporting.py
"""
Functions for calculating performance metrics and plotting results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

def calculate_metrics(final_value, initial_capital, trades_df, portfolio_values, df_index):
    """Calculates and prints key performance metrics."""
    print("\n--- Backtest Results ---")

    start_date = df_index[0] if not df_index.empty else "N/A"
    end_date = df_index[-1] if not df_index.empty else "N/A"
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")

    total_return_percent = ((final_value - initial_capital) / initial_capital) * 100
    print(f"Total Return: {total_return_percent:.2f}%")

    max_drawdown = 0.0

    if not trades_df.empty:
        sell_trades = trades_df[trades_df['Type'].isin(['SELL_EMA', 'STOP_LOSS', 'TAKE_PROFIT'])]
        stop_loss_exits = trades_df[trades_df['Type'] == 'STOP_LOSS']
        take_profit_exits = trades_df[trades_df['Type'] == 'TAKE_PROFIT']
        ema_sell_exits = trades_df[trades_df['Type'] == 'SELL_EMA']

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
        if portfolio_values and not df_index.empty:
            portfolio_index = df_index.copy()
            # Ensure last index is a Timestamp before adding Timedelta
            last_index = portfolio_index[-1]
            if not isinstance(last_index, pd.Timestamp):
                last_index = pd.to_datetime(last_index)
            portfolio_index = portfolio_index.append(pd.Index([last_index + pd.Timedelta(seconds=1)]))

            if len(portfolio_values) == len(portfolio_index):
                portfolio_series = pd.Series(portfolio_values, index=portfolio_index)
                rolling_max = portfolio_series.cummax()
                daily_drawdown = portfolio_series / rolling_max - 1.0
                max_drawdown = daily_drawdown.cummin().min() * 100
                print(f"Maximum Drawdown: {max_drawdown:.2f}%")
            else:
                print(f"Warning: Drawdown calc error - PV length {len(portfolio_values)} vs Index length {len(portfolio_index)}")
                print(f"Maximum Drawdown: N/A")

        else:
             print(f"Maximum Drawdown: N/A (No trades or data)")

    else:
        print("No trades were executed during this period.")
        print(f"Maximum Drawdown: 0.00%") 

    print("-" * 30)


def plot_results(df, portfolio_values, trades_df):
    """Generates the results plot with Price/Indicators, RSI, and Equity Curve."""
    if df.empty:
        print("Cannot plot results: DataFrame is empty.")
        return

    print("\nGenerating plots...")
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [4, 2, 2]})

        # --- Plot 1: Price, EMAs, Trade Signals ---
        axes[0].plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.6, linewidth=1)
        axes[0].plot(df.index, df[config.COL_EMA_SHORT], label=f'{config.EMA_SHORT_PERIOD}-P EMA', color='orange', linewidth=1.5)
        axes[0].plot(df.index, df[config.COL_EMA_LONG], label=f'{config.EMA_LONG_PERIOD}-P EMA', color='red', linewidth=1.5)

        # Plot markers only if trades exist
        if not trades_df.empty:
            valid_timestamps = df.index.intersection(trades_df.index)
            filtered_trades = trades_df.loc[valid_timestamps]

            buy_markers = filtered_trades[filtered_trades['Type'] == 'BUY']
            sell_ema_markers = filtered_trades[filtered_trades['Type'] == 'SELL_EMA']
            stop_loss_markers = filtered_trades[filtered_trades['Type'] == 'STOP_LOSS']
            take_profit_markers = filtered_trades[filtered_trades['Type'] == 'TAKE_PROFIT']

            if not buy_markers.empty:
                axes[0].scatter(buy_markers.index, df.loc[buy_markers.index]['close'], label='Buy', marker='^', color='lime', s=100, alpha=1, zorder=5, edgecolors='black', linewidth=0.5)
            if not sell_ema_markers.empty:
                axes[0].scatter(sell_ema_markers.index, df.loc[sell_ema_markers.index]['close'], label='Sell (EMA)', marker='v', color='red', s=100, alpha=1, zorder=5, edgecolors='black', linewidth=0.5)
            if not stop_loss_markers.empty:
                axes[0].scatter(stop_loss_markers.index, df.loc[stop_loss_markers.index]['low'], label='Stop Loss', marker='x', color='magenta', s=120, alpha=1, zorder=5, linewidth=1.5)
            if not take_profit_markers.empty:
                axes[0].scatter(take_profit_markers.index, df.loc[take_profit_markers.index]['high'], label='Take Profit', marker='*', color='cyan', s=150, alpha=1, zorder=5, edgecolors='black', linewidth=0.5)

        axes[0].set_title(f'{config.SYMBOL} ({config.INTERVAL}) EMA/RSI/ATR Strategy')
        axes[0].set_ylabel('Price (USDT)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)

        # --- Plot 2: RSI ---
        axes[1].plot(df.index, df[config.COL_RSI], label=f'RSI ({config.RSI_PERIOD})', color='purple', linewidth=1)
        axes[1].axhline(config.RSI_OVERBOUGHT, color='red', linestyle='--', linewidth=1, label=f'OB ({config.RSI_OVERBOUGHT})')
        axes[1].axhline(config.RSI_BUY_THRESHOLD, color='blue', linestyle=':', linewidth=1, label=f'Buy Thresh ({config.RSI_BUY_THRESHOLD})')
        axes[1].axhline(config.RSI_SELL_THRESHOLD, color='blue', linestyle=':', linewidth=1, label=f'Sell Thresh ({config.RSI_SELL_THRESHOLD})') # Optional
        axes[1].axhline(30, color='green', linestyle='--', linewidth=1, label='OS (30)')
        axes[1].set_title('Relative Strength Index (RSI)')
        axes[1].set_ylabel('RSI Value')
        axes[1].legend(loc='upper left')
        axes[1].grid(True)

        # --- Plot 3: Equity Curve ---
        if portfolio_values:
             portfolio_index = df.index.copy()
             
             # Ensure portfolio_index is a DatetimeIndex
             if not isinstance(portfolio_index, pd.DatetimeIndex):
                 try:
                     portfolio_index = pd.to_datetime(portfolio_index)
                 except Exception:
                     # Fallback: create a datetime index based on current time
                     portfolio_index = pd.Index([pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(len(portfolio_values))])
             
             if not portfolio_index.empty:
                  portfolio_index = portfolio_index.append(pd.Index([portfolio_index[-1] + pd.Timedelta(seconds=1)]))
             else: 
                 portfolio_index = pd.Index([pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(len(portfolio_values))])

             if len(portfolio_values) == len(portfolio_index):
                  portfolio_series = pd.Series(portfolio_values, index=portfolio_index)
                  axes[2].plot(portfolio_series.index, portfolio_series, label='Portfolio Value', color='green')
             else:
                  axes[2].text(0.5, 0.5, 'Plot Error: PV/Index Length Mismatch', ha='center', va='center', transform=axes[2].transAxes)

        else: 
            axes[2].text(0.5, 0.5, 'No Trades or Data for Equity Curve', ha='center', va='center', transform=axes[2].transAxes)

        axes[2].set_title('Equity Curve')
        axes[2].set_ylabel('Portfolio Value (USDT)')
        axes[2].set_xlabel('Timestamp')
        axes[2].grid(True)
        axes[2].legend(loc='upper left')

        # --- Final Plot Adjustments ---
        fig.autofmt_xdate() 
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
        plt.suptitle(f'Backtest Results: {config.SYMBOL} ({config.INTERVAL})', fontsize=14)
        plt.show()

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()