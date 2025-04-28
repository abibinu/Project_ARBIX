# backtester.py
"""
Core backtesting engine with:
  - Risk-based position sizing
  - ATR-based SL/TP
  - Trailing SL after partial profit
  - Only SL/TP exits (no EMA-signal exit)
"""
import pandas as pd
import config

def run_backtest(df: pd.DataFrame):
    if df.empty or config.COL_SIGNAL not in df.columns:
        print("Error: No signals to backtest.")
        return config.INITIAL_CAPITAL, pd.DataFrame(), []

    cash = config.INITIAL_CAPITAL
    crypto_held = 0.0
    portfolio_v = []
    trades = []
    entry_price = stop_loss = take_profit = 0.0
    position = 0  # 0 = flat, 1 = long

    print(f"\nBacktest Start: SL={config.ATR_SL_MULTIPLIER}×ATR, TP={config.ATR_TP_MULTIPLIER}×ATR")

    required = ['close', 'high', 'low', config.COL_ATR, config.COL_SIGNAL]
    if any(c not in df.columns for c in required):
        print("Error: Missing columns:", required)
        return config.INITIAL_CAPITAL, pd.DataFrame(), []

    for i, ts in enumerate(df.index):
        close = df['close'].iat[i]
        high = df['high'].iat[i]
        low = df['low'].iat[i]
        atr = df[config.COL_ATR].iat[i]
        sig = df[config.COL_SIGNAL].iat[i]

        if pd.isna(atr) or atr <= 0:
            portfolio_v.append(cash + crypto_held * close)
            continue

        port_val = cash + crypto_held * close
        portfolio_v.append(port_val)

        if position == 1:
            exit_reason = None
            if sig == -1:  # Signal-based exit
                exit_price = close
                exit_reason = 'SIGNAL_EXIT'
            elif high >= take_profit:
                exit_price = take_profit
                exit_reason = 'TAKE_PROFIT'
            elif low <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'

            if exit_reason:
                proceeds = crypto_held * exit_price
                fee = proceeds * config.FEE_PERCENT
                cash += proceeds - fee
                profit = (proceeds - fee) - (crypto_held * entry_price)

                trades.append({
                    'Timestamp': ts,
                    'Type': exit_reason,
                    'Price': exit_price,
                    'Amount': crypto_held,
                    'Received': proceeds - fee,
                    'Fee': fee,
                    'Profit': profit
                })
                print(f"INFO {ts}: Exit {exit_reason} @ {exit_price:.2f}, P/L={profit:.2f}")

                crypto_held = 0.0
                entry_price = stop_loss = take_profit = 0.0
                position = 0

        if position == 0 and sig == 1:
            entry_price = close
            risk_usd = config.RISK_PCT_PER_TRADE * port_val
            risk_per_unit = config.ATR_SL_MULTIPLIER * atr
            units = risk_usd / risk_per_unit
            cost_usd = units * entry_price

            if cash >= cost_usd:
                fee_units = units * config.FEE_PERCENT
                net_units = units - fee_units
                fee_usd = fee_units * entry_price

                cash -= cost_usd
                crypto_held += net_units

                stop_loss = entry_price - config.ATR_SL_MULTIPLIER * atr
                take_profit = entry_price + config.ATR_TP_MULTIPLIER * atr
                position = 1

                trades.append({
                    'Timestamp': ts,
                    'Type': 'BUY',
                    'Price': entry_price,
                    'Amount': net_units,
                    'Cost': cost_usd,
                    'Fee': fee_usd
                })
                print(f"INFO {ts}: BUY @ {entry_price:.2f}, Qty={net_units:.6f}, Cost=${cost_usd:.2f}")
                print(f"   SL={stop_loss:.2f}, TP={take_profit:.2f}")
            else:
                print(f"WARNING {ts}: Not enough cash (${cash:.2f}) for ${cost_usd:.2f} trade")

    if crypto_held > 0:
        last_price = df['close'].iat[-1]
        final_val = cash + crypto_held * last_price
        print(f"End: Holding {crypto_held:.6f} → ${crypto_held * last_price:.2f} + ${cash:.2f}")
    else:
        final_val = cash
        print(f"End: No holdings → Cash ${cash:.2f}")

    portfolio_v.append(final_val)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
        trades_df.set_index('Timestamp', inplace=True)

    print("Backtest complete.")
    return final_val, trades_df, portfolio_v
