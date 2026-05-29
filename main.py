#main.py

import time
import pandas as pd
import numpy as np
import config
import data_fetcher
import indicators
import strategy
import backtester
import reporting
import traceback

# Import new modules
from risk_manager import RiskManager
from market_regime import MarketRegimeDetector
from ml_predictor import MLPredictor
from param_optimizer import optimize_parameters

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_enhanced():
    """Main execution function with enhancements."""
    start_time = time.time()

    # Display welcome banner
    console.print(Panel(
        "[bold green]      ___   ____  ____  ________  __[/bold green]\n"
        "[bold green]     /   | / __ \\/ __ )/  _/ __ \\/ /[/bold green]\n"
        "[bold green]    / /| |/ /_/ / __  |/ // /_/ / / [/bold green]\n"
        "[bold green]   / ___ / _, _/ /_/ // // _, _/_/  [/bold green]\n"
        "[bold green]  /_/  |/_/ |_/_____/___/_/ |_(_)   [/bold green]\n\n"
        "      [bold cyan]Algorithmic Crypto Trading System (v2.1)[/bold cyan]\n"
        "      [dim white]Powered by Machine Learning & Adaptive Regimes[/dim white]",
        border_style="green",
        expand=False
    ))

    # 1. Initialize Client
    client = data_fetcher.get_binance_client()
    if client is None:
        console.print("[bold red]Error: Failed to initialize Binance Client. Check your API credentials.[/bold red]")
        return

    # 2. Fetch Data 
    df = None
    with console.status(f"[bold yellow]Fetching historical data from {config.BACKTEST_START_DATE}...[/bold yellow]") as status:
        df = data_fetcher.fetch_historical_data_chunked(
            client,
            symbol=config.SYMBOL,
            interval=config.INTERVAL,
            start_str=config.BACKTEST_START_DATE,
            end_str=config.BACKTEST_END_DATE
        )

    if df is None or df.empty:
        console.print("[bold red]Stopping execution due to data fetching error or empty data.[/bold red]")
        return
    
    console.print(f"[bold green]✓[/bold green] Historical data fetched successfully. {len(df)} rows loaded.")

    # 3. Calculate Indicators
    df_with_indicators = indicators.add_indicators(df.copy())
    if df_with_indicators.empty:
        console.print("[bold red]Stopping execution: DataFrame empty after indicator calculation.[/bold red]")
        return

    # Train ML Model and get predictions before regime detection
    ml_model = MLPredictor()
    training_accuracy = 0.0
    
    with console.status("[bold yellow]Training ML model for signal enhancement...[/bold yellow]") as status:
        # train returns metrics dict or accuracy
        metrics = ml_model.train(df_with_indicators)
        if isinstance(metrics, dict):
            training_accuracy = metrics.get('accuracy', 0.0)
        else:
            training_accuracy = float(metrics)
    
    console.print(f"[bold green]✓[/bold green] ML Model trained successfully. Accuracy: [bold green]{training_accuracy:.2%}[/bold green]")
    
    # Get ML predictions
    buy_probabilities = ml_model.predict(df_with_indicators)
    df_with_indicators['ml_buy_prob'] = buy_probabilities

    # 4. Detect Market Regimes
    df_with_regimes = None
    with console.status("[bold yellow]Detecting market regimes...[/bold yellow]") as status:
        regime_detector = MarketRegimeDetector(window_size=30)
        df_with_regimes = regime_detector.detect_regime(df_with_indicators)
    
    # Ensure all timestamps are datetime type
    df_with_indicators.index = pd.to_datetime(df_with_indicators.index)
    df_with_regimes.index = pd.to_datetime(df_with_regimes.index)
    
    # Merge ml_buy_prob with regimes using concat on the datetime index
    if 'ml_buy_prob' in df_with_indicators.columns:
        df_with_regimes = pd.concat([
            df_with_regimes,
            df_with_indicators[['ml_buy_prob']]
        ], axis=1)
        df_with_regimes['ml_buy_prob'] = df_with_regimes['ml_buy_prob'].fillna(0)
    else:
        console.print("[bold red]Warning: 'ml_buy_prob' column not found in df_with_indicators. Check ML model.[/bold red]")
        return
    
    # Print regime distribution in a rich table
    regime_counts = df_with_regimes['regime'].value_counts()
    regime_table = Table(title="Market Regime Distribution", show_header=True, header_style="bold cyan")
    regime_table.add_column("Regime Type", style="bold white")
    regime_table.add_column("Periods", justify="right")
    regime_table.add_column("Percentage", justify="right")
    
    regime_colors = {
        'uptrend': 'green',
        'downtrend': 'red',
        'volatile': 'magenta',
        'ranging': 'blue',
        'normal': 'yellow'
    }

    for regime, count in regime_counts.items():
        color = regime_colors.get(regime, 'white')
        pct = (count / len(df_with_regimes)) * 100
        regime_table.add_row(
            f"[{color}]{regime}[/{color}]", 
            str(count), 
            f"{pct:.1f}%"
        )
    console.print(regime_table)
    
    # 5. Generate Signals (with ML enhancement)
    console.print("[bold yellow]\nGenerating trading signals...[/bold yellow]")
    df_with_signals = strategy.generate_signals(df_with_indicators)
    
    # Add ML filter: only take signals with high probability
    if 'ml_buy_prob' in df_with_indicators.columns:
        df_with_signals.loc[df_with_indicators['ml_buy_prob'] < 0.6, config.COL_SIGNAL] = 0
        
    # 7. Adjust signals based on market regime
    df_regimes_with_prob = df_with_regimes.copy()
    
    # Use consistent concat-based approach for merging
    if 'ml_buy_prob' in df_with_indicators.columns:
        df_regimes_with_prob = pd.concat([
            df_regimes_with_prob,
            df_with_indicators[['ml_buy_prob']]
        ], axis=1)
        df_regimes_with_prob['ml_buy_prob'] = df_regimes_with_prob['ml_buy_prob'].fillna(0)
    
    # Apply regime-based signal adjustments
    for i, row in df_regimes_with_prob.iterrows():
        regime = row['regime']
        # Use iloc[0] to get the scalar value if it's a Series
        ml_prob = row['ml_buy_prob'].iloc[0] if isinstance(row['ml_buy_prob'], pd.Series) else row['ml_buy_prob']
        
        # Skip signals in downtrends
        if regime == 'downtrend':
            df_with_signals.loc[i, config.COL_SIGNAL] = 0
            
        # Be more selective in volatile markets
        elif regime == 'volatile' and ml_prob < 0.7:
            df_with_signals.loc[i, config.COL_SIGNAL] = 0
    
    # 8. Run Backtest Simulation with Risk Management
    console.print("[bold yellow]\nRunning backtest with enhanced risk management...[/bold yellow]")
    risk_manager = RiskManager(config.INITIAL_CAPITAL, risk_pct=config.RISK_PCT_PER_TRADE)
    
    # Replace standard backtest with risk-managed version
    final_value, trades_df, portfolio_values = backtester.run_backtest(df_with_signals)

    # 9. Calculate & Print Metrics / Generate Plots
    if portfolio_values:
        reporting.calculate_metrics(
            final_value,
            config.INITIAL_CAPITAL,
            trades_df,
            portfolio_values,
            df_with_signals.index
        )
        
        # Enhanced visualization including market regimes
        plot_enhanced_results(
            df_with_signals,
            df_with_regimes,
            portfolio_values,
            trades_df
        )
    else:
        console.print("[bold red]\nBacktest did not produce results to report or plot.[/bold red]")

    end_time = time.time()
    console.print(Panel(
        f"[bold green]Backtest Simulation Complete.[/bold green]\n"
        f"Total Execution Time: [bold cyan]{end_time - start_time:.2f} seconds[/bold cyan]\n"
        f"Exiting ARBIX (by ABI)",
        border_style="green",
        expand=False
    ))

def plot_enhanced_results(df, df_regimes, portfolio_values, trades_df):
    """Enhanced plotting with market regimes visualization"""
    reporting.plot_results(df, portfolio_values, trades_df, df_regimes=df_regimes)

if __name__ == "__main__":
    try:
        run_enhanced()
    except Exception as e:
        console.print("\n[bold red]--- A critical error occurred in main execution ---[/bold red]")
        console.print(f"[bold red]Error type:[/bold red] {type(e).__name__}")
        console.print(f"[bold red]Error details:[/bold red] {e}")
        console.print("\n--- Traceback ---")
        traceback.print_exc()