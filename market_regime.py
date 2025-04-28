# Create a new file called market_regime.py

import pandas as pd
import numpy as np
from scipy import stats

class MarketRegimeDetector:
    def __init__(self, window_size=30):
        """
        Initialize regime detector
        
        window_size: Number of periods to analyze for regime detection
        """
        self.window_size = window_size
        
    def detect_regime(self, df):
        """
        Detect market regime: trending, ranging, or volatile
        
        Returns updated DataFrame with regime column
        """
        df = df.copy()
        
        # Calculate volatility
        log_returns = np.log(df['close'] / df['close'].shift(1))
        volatility = log_returns.rolling(window=self.window_size).std() * np.sqrt(365)
        
        # Ensure volatility and low_vol_threshold are numeric
        volatility = pd.to_numeric(volatility, errors='coerce')
        
        # Calculate trend strength
        price = df['close']
        x = np.arange(len(price))
        
        # Initialize regime column
        df['regime'] = 'unknown'
        
        # Calculate rolling linear regression for trend detection
        df['trend_strength'] = 0
        df['r_squared'] = 0
        
        for i in range(self.window_size, len(df)):
            window_prices = price.iloc[i-self.window_size:i]
            window_x = x[i-self.window_size:i]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(window_x, window_prices)
            
            # Store values
            df.loc[i, 'trend_strength'] = slope
            df.loc[i, 'r_squared'] = r_value**2
        
        # Adaptive thresholds based on quantiles
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)
        
        # Calculate strong trend threshold
        strong_trend_threshold = df['r_squared'].quantile(0.75)
        
        # Add volatility as a column to ensure proper alignment
        df['volatility'] = volatility
        
        # Identify regimes using properly aligned data
        df.loc[(df['r_squared'] > strong_trend_threshold) & (df['trend_strength'] > 0), 'regime'] = 'uptrend'
        df.loc[(df['r_squared'] > strong_trend_threshold) & (df['trend_strength'] < 0), 'regime'] = 'downtrend'
        df.loc[df['volatility'] > high_vol_threshold, 'regime'] = 'volatile'
        df.loc[(df['r_squared'] < 0.3) & (df['volatility'] <= low_vol_threshold), 'regime'] = 'ranging'
        
        # For any remaining 'unknown', classify as normal
        df.loc[df['regime'] == 'unknown', 'regime'] = 'normal'
        
        # Drop the temporary volatility column
        df = df.drop('volatility', axis=1)
        
        return df
    
    def get_optimal_parameters(self, regime):
        """
        Return optimal strategy parameters for current market regime
        """
        if regime == 'uptrend':
            return {
                'EMA_SHORT_PERIOD': 10,    # Faster EMA for stronger trends
                'EMA_LONG_PERIOD': 30,
                'RSI_BUY_THRESHOLD': 45,   # More aggressive entries
                'ATR_SL_MULTIPLIER': 2.0,  # Wider stops for trends
                'ATR_TP_MULTIPLIER': 3.0   # Larger targets to capture trend
            }
        elif regime == 'downtrend':
            return {
                'TRADE_DIRECTION': 'none'  # Avoid trading in downtrends
            }
        elif regime == 'volatile':
            return {
                'EMA_SHORT_PERIOD': 25,    # Slower EMA for noise filtering
                'EMA_LONG_PERIOD': 75,
                'RSI_BUY_THRESHOLD': 60,   # More conservative entries
                'ATR_SL_MULTIPLIER': 1.2,  # Tighter stops
                'ATR_TP_MULTIPLIER': 1.8   # Modest targets
            }
        elif regime == 'ranging':
            return {
                'EMA_SHORT_PERIOD': 15,
                'EMA_LONG_PERIOD': 35,
                'RSI_BUY_THRESHOLD': 40,   # Buy near oversold
                'RSI_SELL_THRESHOLD': 60,  # Sell near overbought
                'ATR_SL_MULTIPLIER': 1.5,
                'ATR_TP_MULTIPLIER': 1.5   # Equal risk:reward
            }
        else:  # normal
            return {
                'EMA_SHORT_PERIOD': 20,
                'EMA_LONG_PERIOD': 50,
                'RSI_BUY_THRESHOLD': 55,
                'ATR_SL_MULTIPLIER': 1.5,
                'ATR_TP_MULTIPLIER': 2.0
            }