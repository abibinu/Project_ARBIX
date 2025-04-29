# Create a new file called risk_manager.py

import pandas as pd
import numpy as np
import config

class RiskManager:
    def __init__(self, portfolio_value, risk_pct=0.01):
        self.portfolio_value = portfolio_value
        self.risk_pct = risk_pct
        self.in_drawdown = False
        self.peak_value = portfolio_value
        self.drawdown_level = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.trade_history = []
        
    def update_portfolio_value(self, new_value):
        """Update portfolio value and track drawdown"""
        self.portfolio_value = new_value
        
        # Update peak value
        if new_value > self.peak_value:
            self.peak_value = new_value
            self.in_drawdown = False
        
        # Calculate current drawdown
        self.drawdown_level = 1 - (new_value / self.peak_value)
        
        # Check if in drawdown
        if self.drawdown_level > 0.05:  # 5% drawdown
            self.in_drawdown = True
        
        # Add a log for drawdown level
        print(f"Current drawdown level: {self.drawdown_level:.2%}")
        
        return self.in_drawdown, self.drawdown_level
    
    def record_trade(self, trade_result):
        """Record trade result and update metrics"""
        self.trade_history.append(trade_result)
        
        if trade_result < 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses, self.consecutive_losses
            )
        else:
            self.consecutive_losses = 0
    
    def get_position_size(self, entry_price, stop_loss):
        """
        Calculate position size based on risk management rules
        
        Dynamic risk adjustment based on:
        - Current drawdown
        - Recent win/loss streak
        - Market volatility
        """
        risk_amount = self.portfolio_value * self.risk_pct
        
        # Risk adjustment based on drawdown
        if self.drawdown_level > 0.15:  # Increased from 0.1
            risk_amount *= 0.5  # Reduce risk by half
        elif self.drawdown_level > 0.10:  # Increased from 0.05
            risk_amount *= 0.75  # Reduce risk by 25%
            
        # Risk reduction after consecutive losses
        if self.consecutive_losses >= 2:  # Reduced from 3 for tighter control
            risk_amount *= 0.5  # Reduce risk by half after 3 consecutive losses
            
        # Calculate position size
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0  # Avoid division by zero
            
        position_size = risk_amount / risk_per_unit
        
        return position_size
        
    def should_trade(self, market_condition='normal'):
        """
        Determine if trading should be allowed based on risk metrics
        
        market_condition: 'normal', 'volatile', 'trending'
        """
        # No trading during extreme drawdowns
        if self.drawdown_level > 0.25:  # Increased from 0.2 to 0.25
            return False
            
        # Reduce trading frequency after consecutive losses
        if self.consecutive_losses >= 3:  # Reduced from 5 to 3 for tighter control
            return False
            
        # Market-specific adjustments
        if market_condition == 'volatile' and self.drawdown_level > 0.15:  # Increased from 0.1
            return False
            
        return True