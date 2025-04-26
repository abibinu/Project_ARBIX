# backtester.py
"""
Core backtesting simulation engine.
"""
import pandas as pd
import config 

def run_backtest(df_with_signals):
    """Runs the backtesting simulation loop."""
    if df_with_signals.empty or config.COL_SIGNAL not in df_with_signals.columns:
         print("Error: DataFrame is empty or signals not generated. Cannot run backtest.")
         return config.INITIAL_CAPITAL, pd.DataFrame(), [] 

    print(f"\nStarting backtesting simulation with dynamic ATR SL/TP...")
    print(f"ATR Multipliers: SL={config.ATR_SL_MULTIPLIER}, TP={config.ATR_TP_MULTIPLIER}")

    # Initialize variables
    cash = config.INITIAL_CAPITAL
    crypto_held = 0.0
    portfolio_values = []
    trades = []
    entry_price = 0.0
    stop_loss_level = 0.0
    take_profit_level = 0.0
    position_status = 0 

    # Ensure required columns exist
    required_cols = ['close', 'low', 'high', config.COL_ATR, config.COL_SIGNAL]
    if not all(col in df_with_signals.columns for col in required_cols):
        print(f"Error: Missing required columns in DataFrame for backtest: {required_cols}")
        return config.INITIAL_CAPITAL, pd.DataFrame(), []

    # Simulation loop
    for i in range(len(df_with_signals)):
        current_index = df_with_signals.index[i]
        current_close = df_with_signals['close'].iloc[i]
        current_low = df_with_signals['low'].iloc[i]
        current_high = df_with_signals['high'].iloc[i]
        current_atr = df_with_signals[config.COL_ATR].iloc[i]
        actual_signal_this_bar = df_with_signals[config.COL_SIGNAL].iloc[i]

        # Handle potential invalid ATR
        if pd.isna(current_atr) or current_atr <= 0: 
             print(f"Warning {current_index}: Skipping bar due to invalid ATR ({current_atr})")
             portfolio_values.append(cash + (crypto_held * current_close))
             continue

        # Append portfolio value *before* potential trades for this bar
        current_portfolio_value = cash + (crypto_held * current_close)
        portfolio_values.append(current_portfolio_value)

        exit_reason = None

        # --- Check Exit Conditions (Only if in a long position) ---
        if position_status == 1:
            # Check TP first
            if current_high >= take_profit_level:
                exit_price = take_profit_level
                exit_reason = 'TAKE_PROFIT'
                print(f"INFO {current_index}: TAKE-PROFIT triggered at {exit_price:.2f} (Entry: {entry_price:.2f}, High: {current_high:.2f})")
            # Check SL second
            elif current_low <= stop_loss_level:
                exit_price = stop_loss_level
                exit_reason = 'STOP_LOSS'
                print(f"INFO {current_index}: STOP-LOSS triggered at {exit_price:.2f} (Entry: {entry_price:.2f}, Low: {current_low:.2f})")
            # Check EMA signal third
            elif actual_signal_this_bar == -1:
                exit_price = current_close
                exit_reason = 'SELL_EMA'
                print(f"INFO {current_index}: EMA SELL signal -> Selling @ {exit_price:.2f} (Entry: {entry_price:.2f})")

            # --- Execute Exit if any condition met ---
            if exit_reason:
                sell_value_usd = crypto_held * exit_price
                fee = sell_value_usd * config.FEE_PERCENT
                cash += (sell_value_usd - fee)
                sell_amount_crypto = crypto_held
                profit = (sell_value_usd - fee) - (sell_amount_crypto * entry_price)
                trades.append({'Timestamp': current_index, 'Type': exit_reason, 'Price': exit_price, 'Amount_Crypto': sell_amount_crypto, 'Received_USD': sell_value_usd - fee, 'Fee': fee, 'Profit': profit})
                # Reset position state
                crypto_held = 0.0
                entry_price = 0.0
                stop_loss_level = 0.0
                take_profit_level = 0.0
                position_status = 0

        # --- Check Entry Condition (Only if neutral) ---
        # Important: Check position status *after* potential exit from the *same* bar
        if position_status == 0:
            if actual_signal_this_bar == 1:
                entry_price = current_close
                buy_amount_crypto = config.TRADE_AMOUNT_USD / entry_price
                fee_crypto = buy_amount_crypto * config.FEE_PERCENT
                buy_amount_crypto_net = buy_amount_crypto - fee_crypto
                fee_usd = fee_crypto * entry_price

                if cash >= config.TRADE_AMOUNT_USD: # Check if enough cash for trade
                    cash -= config.TRADE_AMOUNT_USD
                    crypto_held += buy_amount_crypto_net
                    # Set dynamic SL and TP
                    stop_loss_level = entry_price - (config.ATR_SL_MULTIPLIER * current_atr)
                    take_profit_level = entry_price + (config.ATR_TP_MULTIPLIER * current_atr)
                    position_status = 1
                    trades.append({'Timestamp': current_index, 'Type': 'BUY', 'Price': entry_price, 'Amount_Crypto': buy_amount_crypto_net, 'Cost_USD': config.TRADE_AMOUNT_USD, 'Fee': fee_usd})
                    print(f"INFO {current_index}: BUY signal -> Bought {buy_amount_crypto_net:.6f} @ {entry_price:.2f}")
                    print(f"   SL set at {stop_loss_level:.2f}, TP set at {take_profit_level:.2f} (ATR: {current_atr:.2f})")
                else:
                    print(f"Warning {current_index}: Insufficient cash ({cash:.2f}) to execute BUY trade costing {config.TRADE_AMOUNT_USD:.2f}")


        # --- Post-Loop Calculations ---
    # Calculate final portfolio value after the last bar processed
    final_value = cash + (crypto_held * df_with_signals['close'].iloc[-1] if crypto_held > 0 and not df_with_signals.empty else cash) # Use last close if holding

    if not df_with_signals.empty: # Only append if data was processed
        portfolio_values.append(final_value)

    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
         trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
         trades_df = trades_df.set_index('Timestamp')

    print("Backtesting simulation finished.")
    return final_value, trades_df, portfolio_values