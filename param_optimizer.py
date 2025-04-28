# Add to your project as param_optimizer.py

import pandas as pd
import numpy as np
from itertools import product
import config
import indicators
import strategy
import backtester
import copy
import matplotlib.pyplot as plt

def optimize_parameters(df, param_ranges):
    """
    Optimize strategy parameters using grid search
    
    param_ranges: Dict of parameter names and lists of values to test
    Returns: Dict of best parameters and their performance metrics
    """
    results = []
    best_return = -float('inf')
    best_params = None
    best_sharpe = -float('inf')
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(product(*param_ranges.values()))
    
    print(f"Testing {len(param_values)} parameter combinations...")
    
    for i, values in enumerate(param_values):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(param_values)} combinations tested")
            
        # Create temp config with these parameters
        temp_config = copy.deepcopy(config)
        for name, value in zip(param_names, values):
            setattr(temp_config, name, value)
        
        # Update derived parameters
        temp_config.COL_EMA_SHORT = f'EMA_{temp_config.EMA_SHORT_PERIOD}'
        temp_config.COL_EMA_LONG = f'EMA_{temp_config.EMA_LONG_PERIOD}'
        temp_config.COL_RSI = f'RSI_{temp_config.RSI_PERIOD}'
        temp_config.COL_ATR = f'ATRr_{temp_config.ATR_PERIOD}'
        
        # Calculate indicators with these parameters
        df_temp = df.copy()
        df_with_indicators = indicators.add_indicators_with_config(df_temp, temp_config)
        
        # Generate signals using temp config
        df_with_signals = strategy.generate_signals_with_config(df_with_indicators, temp_config)
        
        # Run backtest
        final_value, trades_df, portfolio_values = backtester.run_backtest_with_config(
            df_with_signals, temp_config, verbose=False)
        
        # Calculate metrics
        total_return = ((final_value - temp_config.INITIAL_CAPITAL) / 
                         temp_config.INITIAL_CAPITAL) * 100
        
        # Calculate Sharpe ratio
        if portfolio_values and len(portfolio_values) > 1:
            returns = [portfolio_values[i]/portfolio_values[i-1]-1 for i in range(1, len(portfolio_values))]
            if returns:
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
            
        # Count trades
        num_trades = len(trades_df) // 2 if not trades_df.empty else 0
        
        # Store results
        result = {
            'params': dict(zip(param_names, values)),
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'num_trades': num_trades
        }
        results.append(result)
        
        # Track best parameters by return
        if total_return > best_return:
            best_return = total_return
            best_params = dict(zip(param_names, values))
        
        # Also track best by Sharpe
        if sharpe > best_sharpe and num_trades >= 5:  # Require minimum trades
            best_sharpe = sharpe
            best_sharpe_params = dict(zip(param_names, values))
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot return distribution
    axes[0].hist(results_df['total_return'], bins=30)
    axes[0].set_title('Distribution of Total Returns')
    axes[0].set_xlabel('Total Return %')
    axes[0].set_ylabel('Frequency')
    
    # Plot Sharpe ratio distribution
    axes[1].hist(results_df['sharpe_ratio'], bins=30)
    axes[1].set_title('Distribution of Sharpe Ratios')
    axes[1].set_xlabel('Sharpe Ratio')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Find top 10 best performing parameter sets
    top_by_return = results_df.sort_values('total_return', ascending=False).head(10)
    top_by_sharpe = results_df.sort_values('sharpe_ratio', ascending=False).head(10)
    
    print("\nTop 10 Parameter Sets by Return:")
    print(top_by_return[['params', 'total_return', 'sharpe_ratio', 'num_trades']])
    
    print("\nTop 10 Parameter Sets by Sharpe Ratio:")
    print(top_by_sharpe[['params', 'total_return', 'sharpe_ratio', 'num_trades']])
    
    print(f"\nBest parameters by return: {best_params}")
    print(f"Best parameters by Sharpe: {best_sharpe_params}")
    
    return {
        'best_params_return': best_params,
        'best_params_sharpe': best_sharpe_params,
        'results_df': results_df
    }

# Example parameter ranges to test
example_param_ranges = {
    'EMA_SHORT_PERIOD': [10, 15, 20, 25],
    'EMA_LONG_PERIOD': [40, 50, 60],
    'RSI_PERIOD': [9, 14],
    'RSI_BUY_THRESHOLD': [45, 50, 55, 60],
    'ATR_SL_MULTIPLIER': [1.0, 1.5, 2.0],
    'ATR_TP_MULTIPLIER': [1.5, 2.0, 2.5]
}