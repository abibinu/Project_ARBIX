# reporting.py
"""
Functions for calculating performance metrics and plotting results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

console = Console()

def calculate_metrics(final_value, initial_capital, trades_df, portfolio_values, df_index):
    """Calculates and prints key performance metrics using rich tables."""
    start_date = df_index[0] if not df_index.empty else "N/A"
    end_date = df_index[-1] if not df_index.empty else "N/A"
    
    # Header panel
    console.print(Panel(
        f"[bold cyan]Simulation Period:[/bold cyan] {start_date} to {end_date}\n"
        f"[bold cyan]Initial Capital  :[/bold cyan] ${initial_capital:.2f}\n"
        f"[bold cyan]Final Value      :[/bold cyan] ${final_value:.2f}",
        title="[bold yellow]ARBIX Backtest Results[/bold yellow]",
        border_style="yellow"
    ))

    total_return_percent = ((final_value - initial_capital) / initial_capital) * 100
    
    # 1. Performance Overview Table
    perf_table = Table(title="Performance Summary", show_header=True, header_style="bold magenta")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", justify="right")
    
    return_style = "bold green" if total_return_percent >= 0 else "bold red"
    perf_table.add_row("Total Return", f"[{return_style}]{total_return_percent:.2f}%[/{return_style}]")
    
    max_drawdown = 0.0

    if not trades_df.empty:
        sell_trades = trades_df[trades_df['Type'].isin(['SELL_EMA', 'STOP_LOSS', 'TAKE_PROFIT'])]
        stop_loss_exits = trades_df[trades_df['Type'] == 'STOP_LOSS']
        take_profit_exits = trades_df[trades_df['Type'] == 'TAKE_PROFIT']
        ema_sell_exits = trades_df[trades_df['Type'] == 'SELL_EMA']

        # Calculate Win Rate
        win_rate = 0.0
        total_profit = 0.0
        total_loss = 0.0
        profit_factor = np.inf

        if not sell_trades.empty:
            profitable_trades = sell_trades[sell_trades['Profit'] > 0]
            loss_trades = sell_trades[sell_trades['Profit'] <= 0]
            win_rate = (len(profitable_trades) / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
            
            total_profit = profitable_trades['Profit'].sum()
            total_loss = abs(loss_trades['Profit'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

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
                drawdown_str = f"{max_drawdown:.2f}%"
            else:
                drawdown_str = "Error (index mismatch)"
        else:
            drawdown_str = "N/A"

        perf_table.add_row("Maximum Drawdown", f"[bold red]{drawdown_str}[/bold red]")
        console.print(perf_table)

        # 2. Detailed Trade Statistics Table
        trade_table = Table(title="Execution & Trade Statistics", show_header=True, header_style="bold magenta")
        trade_table.add_column("Statistic", style="cyan")
        trade_table.add_column("Value", justify="right")
        
        trade_table.add_row("Total Trades Executed (Closed)", f"{len(sell_trades)}")
        trade_table.add_row("Stop-Loss Exits", f"[red]{len(stop_loss_exits)}[/red]")
        trade_table.add_row("Take-Profit Exits", f"[green]{len(take_profit_exits)}[/green]")
        trade_table.add_row("EMA Crossover Exits", f"[blue]{len(ema_sell_exits)}[/blue]")
        
        win_rate_style = "bold green" if win_rate >= 50 else "bold yellow"
        trade_table.add_row("Win Rate", f"[{win_rate_style}]{win_rate:.2f}%[/{win_rate_style}]")
        trade_table.add_row("Total Winner Profit", f"[green]${total_profit:.2f}[/green]")
        trade_table.add_row("Total Loser Loss", f"[red]${total_loss:.2f}[/red]")
        
        pf_style = "bold green" if profit_factor >= 1.5 else ("bold yellow" if profit_factor >= 1.0 else "bold red")
        pf_str = f"{profit_factor:.2f}" if profit_factor != np.inf else "∞"
        trade_table.add_row("Profit Factor", f"[{pf_style}]{pf_str}[/{pf_style}]")

        console.print(trade_table)
    else:
        perf_table.add_row("Maximum Drawdown", "0.00%")
        console.print(perf_table)
        console.print("[yellow]\nNo trades were executed during this period.[/yellow]")

    console.print("[bold yellow]" + "-" * 40 + "[/bold yellow]")


def plot_results(df, portfolio_values, trades_df, df_regimes=None):
    """Generates a TradingView-inspired dark results plot with Price/Indicators, RSI, and Equity Curve."""
    if df.empty:
        console.print("[bold red]Cannot plot results: DataFrame is empty.[/bold red]")
        return

    console.print("\n[bold yellow]Generating professional plots...[/bold yellow]")
    try:
        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Set up modern style params manually to avoid relying on seaborn styles
        plt.rcParams['figure.facecolor'] = '#131722'
        plt.rcParams['axes.facecolor'] = '#1c2030'
        plt.rcParams['text.color'] = '#d1d4dc'
        plt.rcParams['axes.labelcolor'] = '#d1d4dc'
        plt.rcParams['xtick.color'] = '#707a8a'
        plt.rcParams['ytick.color'] = '#707a8a'
        plt.rcParams['axes.edgecolor'] = '#2a2e39'
        plt.rcParams['grid.color'] = '#2a2e39'
        plt.rcParams['grid.alpha'] = 0.5

        # Create figure with proper spacing
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [4, 2, 2]})
        
        # --- Plot 1: Price, EMAs, Trade Signals ---
        axes[0].plot(df.index, df['close'], label='Close Price', color='#2962ff', alpha=0.8, linewidth=1)
        axes[0].plot(df.index, df[config.COL_EMA_SHORT], label=f'{config.EMA_SHORT_PERIOD}-P EMA', color='#ff9800', linewidth=1.2, alpha=0.9)
        axes[0].plot(df.index, df[config.COL_EMA_LONG], label=f'{config.EMA_LONG_PERIOD}-P EMA', color='#f44336', linewidth=1.2, alpha=0.9)

        # Plot markers only if trades exist
        if not trades_df.empty:
            # Ensure trades_df index is datetime
            if not isinstance(trades_df.index, pd.DatetimeIndex):
                trades_df.index = pd.to_datetime(trades_df.index)
                
            valid_timestamps = df.index.intersection(trades_df.index)
            filtered_trades = trades_df.loc[valid_timestamps]

            buy_markers = filtered_trades[filtered_trades['Type'] == 'BUY']
            sell_ema_markers = filtered_trades[filtered_trades['Type'] == 'SELL_EMA']
            stop_loss_markers = filtered_trades[filtered_trades['Type'] == 'STOP_LOSS']
            take_profit_markers = filtered_trades[filtered_trades['Type'] == 'TAKE_PROFIT']

            if not buy_markers.empty:
                axes[0].scatter(buy_markers.index, df.loc[buy_markers.index]['close'], 
                              label='Buy Signal', marker='^', color='#26a69a', s=120, alpha=1, 
                              zorder=5, edgecolors='#ffffff', linewidth=1.0)
            if not sell_ema_markers.empty:
                axes[0].scatter(sell_ema_markers.index, df.loc[sell_ema_markers.index]['close'], 
                              label='Sell (EMA)', marker='v', color='#ef5350', s=120, alpha=1, 
                              zorder=5, edgecolors='#ffffff', linewidth=1.0)
            if not stop_loss_markers.empty:
                axes[0].scatter(stop_loss_markers.index, df.loc[stop_loss_markers.index]['low'], 
                              label='Stop Loss Hit', marker='x', color='#e91e63', s=130, alpha=1, 
                              zorder=6, linewidth=2.0)
            if not take_profit_markers.empty:
                axes[0].scatter(take_profit_markers.index, df.loc[take_profit_markers.index]['high'], 
                              label='Take Profit Hit', marker='*', color='#00e676', s=180, alpha=1, 
                              zorder=6, edgecolors='#ffffff', linewidth=0.8)

        axes[0].set_title(f'{config.SYMBOL} ({config.INTERVAL}) Technical Analysis', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price (USDT)')
        legend0 = axes[0].legend(loc='upper left', facecolor='#131722', edgecolor='#2a2e39')
        plt.setp(legend0.get_texts(), color='#d1d4dc')
        axes[0].grid(True, linestyle='--', linewidth=0.5)

        # --- Plot 2: RSI ---
        axes[1].plot(df.index, df[config.COL_RSI], label=f'RSI ({config.RSI_PERIOD})', color='#b388ff', linewidth=1.2)
        axes[1].axhline(config.RSI_OVERBOUGHT, color='#ff5252', linestyle='--', linewidth=1, label=f'OB ({config.RSI_OVERBOUGHT})')
        axes[1].axhline(config.RSI_BUY_THRESHOLD, color='#2979ff', linestyle=':', linewidth=0.8, label=f'Buy Thresh ({config.RSI_BUY_THRESHOLD})')
        axes[1].axhline(config.RSI_SELL_THRESHOLD, color='#2979ff', linestyle=':', linewidth=0.8, label=f'Sell Thresh ({config.RSI_SELL_THRESHOLD})')
        axes[1].axhline(30, color='#00e676', linestyle='--', linewidth=1, label='OS (30)')
        
        # Shade RSI boundary region
        axes[1].fill_between(df.index, 30, config.RSI_OVERBOUGHT, color='#b388ff', alpha=0.03)

        axes[1].set_title('Relative Strength Index (RSI)', fontsize=11)
        axes[1].set_ylabel('RSI Value')
        axes[1].set_ylim(0, 100)
        legend1 = axes[1].legend(loc='upper left', facecolor='#131722', edgecolor='#2a2e39', ncol=2)
        plt.setp(legend1.get_texts(), color='#d1d4dc')
        axes[1].grid(True, linestyle='--', linewidth=0.5)

        # --- Plot 3: Equity Curve ---
        if portfolio_values and len(portfolio_values) > 0:
            if len(portfolio_values) > len(df.index):
                portfolio_values = portfolio_values[:len(df.index)]
            elif len(portfolio_values) < len(df.index):
                last_value = portfolio_values[-1]
                portfolio_values.extend([last_value] * (len(df.index) - len(portfolio_values)))

            portfolio_series = pd.Series(portfolio_values, index=df.index)
            axes[2].plot(portfolio_series.index, portfolio_series, label='Portfolio Value (Equity)', color='#26a69a', linewidth=1.5)
            
            # Shaded fill under equity curve
            axes[2].fill_between(portfolio_series.index, portfolio_series, config.INITIAL_CAPITAL, 
                                 where=(portfolio_series >= config.INITIAL_CAPITAL), interpolate=True, color='#26a69a', alpha=0.1)
            axes[2].fill_between(portfolio_series.index, portfolio_series, config.INITIAL_CAPITAL, 
                                 where=(portfolio_series < config.INITIAL_CAPITAL), interpolate=True, color='#ef5350', alpha=0.1)

            axes[2].set_title('Portfolio Valuation (Equity Curve)', fontsize=11)
            axes[2].set_ylabel('Value (USDT)')
            legend2 = axes[2].legend(loc='upper left', facecolor='#131722', edgecolor='#2a2e39')
            plt.setp(legend2.get_texts(), color='#d1d4dc')
        else:
            axes[2].text(0.5, 0.5, 'No Trades or Data for Equity Curve', ha='center', va='center', transform=axes[2].transAxes)

        axes[2].grid(True, linestyle='--', linewidth=0.5)

        # --- Plot 4: Market Regimes Shading (if provided) ---
        if df_regimes is not None and 'regime' in df_regimes.columns:
            # Align regimes
            df_regimes_aligned = df_regimes.loc[df_regimes.index.intersection(df.index)]
            if not df_regimes_aligned.empty:
                regime_series = df_regimes_aligned['regime']
                # Detect changes
                changes = (regime_series != regime_series.shift(1))
                change_indices = regime_series.index[changes].tolist() + [regime_series.index[-1]]
                
                # Regime translucent color schemes
                regime_colors = {
                    'uptrend': '#4caf50',   # Soft green
                    'downtrend': '#f44336', # Soft red
                    'volatile': '#9c27b0',  # Soft purple
                    'ranging': '#607d8b',   # Soft blue-gray
                    'normal': '#131722'     # Default dark background
                }
                
                # Highlight backgrounds
                for idx in range(len(change_indices) - 1):
                    start_t = change_indices[idx]
                    end_t = change_indices[idx + 1]
                    reg = regime_series.loc[start_t]
                    if isinstance(reg, pd.Series):
                        reg = reg.iloc[0]
                    
                    if reg in regime_colors and reg != 'normal':
                        color = regime_colors[reg]
                        # Draw span in price plot and equity plot
                        axes[0].axvspan(start_t, end_t, color=color, alpha=0.07, zorder=0)
                        axes[2].axvspan(start_t, end_t, color=color, alpha=0.07, zorder=0)

        # Rotate and align the tick labels
        for ax in axes:
            ax.tick_params(axis='x', rotation=30)
            
        # Use AutoDateFormatter for better date display
        from matplotlib.dates import AutoDateFormatter, AutoDateLocator
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        axes[2].xaxis.set_major_locator(locator)
        axes[2].xaxis.set_major_formatter(formatter)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f'ARBIX Backtest Analytics: {config.SYMBOL} ({config.INTERVAL})', fontsize=14, fontweight='bold', y=0.98, color='#ffffff')
        
        # Save plot configuration
        plt.show()

    except Exception as e:
        console.print(f"[bold red]Error generating plots:[/bold red] {e}")
        import traceback
        traceback.print_exc()


def generate_dashboard_layout(trader, current_price, next_update_s, mode="Paper"):
    """Generates a beautiful dynamic dashboard layout for running bot."""
    # Header
    header_content = Text.assemble(
        ("ARBIX CRYPTO BOT  ", "bold green"),
        (f"|  MODE: {mode.upper()}  ", "bold cyan"),
        (f"|  SYMBOL: {config.SYMBOL}  ", "bold yellow"),
        (f"|  TIMEFRAME: {config.INTERVAL}  ", "bold magenta"),
        (f"|  STATUS: ACTIVE", "bold blink green")
    )
    header = Panel(Align.center(header_content), border_style="green")
    
    # Left Panel: Market & Strategy
    market_table = Table.grid(padding=(0, 1))
    market_table.add_column(style="bold white", width=18)
    market_table.add_column(style="cyan")
    
    market_table.add_row("Current Price:", f"${current_price:.4f}" if current_price else "Fetching...")
    
    regime = getattr(trader, 'current_regime', 'unknown')
    regime_colors = {
        'uptrend': '[bold green]uptrend[/bold green]',
        'downtrend': '[bold red]downtrend[/bold red]',
        'volatile': '[bold magenta]volatile[/bold magenta]',
        'ranging': '[bold blue]ranging[/bold blue]',
        'normal': '[bold yellow]normal[/bold yellow]'
    }
    market_table.add_row("Market Regime:", regime_colors.get(regime, regime))
    
    ml_prob = getattr(trader, 'last_ml_prob', 0.5)
    ml_style = "green" if ml_prob >= config.ML_CONFIDENCE_THRESHOLD else "red"
    market_table.add_row("ML Buy Prob:", f"[{ml_style}]{ml_prob:.2%}[/{ml_style}]")
    market_table.add_row("Next Update in:", f"{next_update_s}s" if next_update_s is not None else "Updating...")
    
    left_panel = Panel(market_table, title="[bold yellow]Market & Strategy[/bold yellow]", border_style="cyan")
    
    # Right Panel: Account & Balance
    account_table = Table.grid(padding=(0, 1))
    account_table.add_column(style="bold white", width=18)
    account_table.add_column(style="green")
    
    initial = config.INITIAL_CAPITAL
    current_bal = trader.balance
    total_pnl = current_bal - initial
    pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0.0
    pnl_style = "bold green" if total_pnl >= 0 else "bold red"
    pnl_sign = "+" if total_pnl >= 0 else ""
    
    account_table.add_row("Initial Balance:", f"${initial:.4f}")
    account_table.add_row("Current Balance:", f"${current_bal:.4f}")
    account_table.add_row("Net Profit/Loss:", f"[{pnl_style}]{pnl_sign}${total_pnl:.4f} ({pnl_sign}{pnl_pct:.2f}%)[/{pnl_style}]")
    account_table.add_row("Max Trades Limit:", f"{config.MAX_TRADES}")
    
    right_panel = Panel(account_table, title="[bold yellow]Account & Balance[/bold yellow]", border_style="cyan")
    
    # Combined Top Row
    top_grid = Table.grid(expand=True)
    top_grid.add_column(ratio=1)
    top_grid.add_column(ratio=1)
    top_grid.add_row(left_panel, right_panel)
    
    # Middle Panel: Active Positions
    pos_table = Table(show_header=True, header_style="bold magenta", expand=True)
    pos_table.add_column("Symbol", style="cyan")
    pos_table.add_column("Side", style="bold white")
    pos_table.add_column("Entry Price", justify="right")
    pos_table.add_column("Current Price", justify="right")
    pos_table.add_column("Quantity", justify="right")
    pos_table.add_column("Stop Loss", justify="right", style="red")
    pos_table.add_column("Take Profit", justify="right", style="green")
    pos_table.add_column("Unrealized PnL", justify="right")
    
    if trader.positions:
        for sym, pos in trader.positions.items():
            pos_pnl = pos.pnl
            # calculate entry cost
            entry_cost = pos.entry_price * pos.quantity
            pos_pnl_pct = (pos_pnl / entry_cost) * 100 if entry_cost > 0 else 0.0
            pos_pnl_style = "bold green" if pos_pnl >= 0 else "bold red"
            pos_pnl_sign = "+" if pos_pnl >= 0 else ""
            
            pos_table.add_row(
                sym,
                pos.side.upper(),
                f"${pos.entry_price:.4f}",
                f"${current_price:.4f}" if current_price else "N/A",
                f"{pos.quantity:.4f}",
                f"${pos.stop_loss:.4f}",
                f"${pos.take_profit:.4f}",
                f"[{pos_pnl_style}]{pos_pnl_sign}${pos_pnl:.4f} ({pos_pnl_sign}{pos_pnl_pct:.2f}%)[/{pos_pnl_style}]"
            )
    else:
        pos_table.add_row("No active positions", "-", "-", "-", "-", "-", "-", "-")
        
    pos_panel = Panel(pos_table, title="[bold yellow]Active Positions[/bold yellow]", border_style="magenta")
    
    # Bottom Panel: Logs / Actions
    log_text = Text()
    trader_logs = getattr(trader, 'logs', [])
    recent_logs = trader_logs[-6:]
    if not recent_logs:
        log_text.append("Waiting for activity logs...", style="dim white")
    else:
        for log_line in recent_logs:
            line_text = Text(log_line + "\n")
            line_text.highlight_words(["BUY", "Opened", "Opened long", "Opened short"], "bold green")
            line_text.highlight_words(["Closed", "Exit", "STOP_LOSS", "TAKE_PROFIT"], "bold red")
            line_text.highlight_words(["Error", "Exception"], "bold red reverse")
            line_text.highlight_words(["Warning", "WARNING"], "bold yellow")
            log_text.append(line_text)
            
    log_panel = Panel(log_text, title="[bold yellow]Recent Activity Log[/bold yellow]", border_style="yellow")
    
    return Group(header, top_grid, pos_panel, log_panel)