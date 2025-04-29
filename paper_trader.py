"""
Trading implementation that handles both paper and real trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional

import live_config as config
import data_fetcher  # Changed to import the full module
from indicators import add_indicators
from strategy import generate_signals
from risk_manager import RiskManager
from market_regime import MarketRegimeDetector
from ml_predictor import MLPredictor
from notifications import TelegramNotifier
from trade_executor import TradeExecutor

class Position:
    def __init__(self, symbol: str, entry_price: float, quantity: float, 
                 side: str, stop_loss: float, take_profit: float):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.side = side  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now()
        self.trailing_stop = None
        self.max_price = entry_price if side == 'long' else float('inf')
        self.min_price = entry_price if side == 'short' else float('-inf')
        self.pnl = 0.0
        self.status = 'open'
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None

    def update(self, current_price: float) -> Optional[dict]:
        """Update position and check for exit conditions"""
        if self.status != 'open':
            return None

        # Update max/min prices for trailing stop
        if self.side == 'long':
            self.max_price = max(self.max_price, current_price)
        else:
            self.min_price = min(self.min_price, current_price)

        # Calculate unrealized PnL
        self.pnl = self._calculate_pnl(current_price)

        # Check for trailing stop activation
        if config.TRAILING_STOP and self.pnl > config.TRAIL_AFTER_PROFIT:
            if self.side == 'long':
                self.trailing_stop = self.max_price * (1 - config.TRAIL_STOP_DISTANCE)
            else:
                self.trailing_stop = self.min_price * (1 + config.TRAIL_STOP_DISTANCE)

        # Check exit conditions
        exit_signal = None
        if self.side == 'long':
            if current_price <= self.stop_loss:
                exit_signal = {'reason': 'stop_loss', 'price': current_price}
            elif current_price >= self.take_profit:
                exit_signal = {'reason': 'take_profit', 'price': current_price}
            elif self.trailing_stop and current_price <= self.trailing_stop:
                exit_signal = {'reason': 'trailing_stop', 'price': current_price}
        else:  # short position
            if current_price >= self.stop_loss:
                exit_signal = {'reason': 'stop_loss', 'price': current_price}
            elif current_price <= self.take_profit:
                exit_signal = {'reason': 'take_profit', 'price': current_price}
            elif self.trailing_stop and current_price >= self.trailing_stop:
                exit_signal = {'reason': 'trailing_stop', 'price': current_price}

        return exit_signal

    def close(self, exit_price: float, reason: str):
        """Close the position"""
        self.status = 'closed'
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.pnl = self._calculate_pnl(exit_price)

    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate position PnL"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def to_dict(self) -> dict:
        """Convert position to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'side': self.side,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'trailing_stop': self.trailing_stop,
            'max_price': self.max_price,
            'min_price': self.min_price,
            'pnl': self.pnl,
            'status': self.status,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason
        }

class PaperTrader:
    def __init__(self):
        self.client = data_fetcher.get_binance_client()  # Updated to use module
        self.positions: Dict[str, Position] = {}
        self.trade_executor = TradeExecutor(self.client) if not config.PAPER_TRADING else None
        
        # Get actual balance from exchange or use initial capital for paper trading
        if not config.PAPER_TRADING and self.trade_executor:
            self.balance = self.trade_executor.get_balance('USDT')
            print(f"Fetched actual balance from {'Testnet' if config.USE_TESTNET else 'Live'}: ${self.balance:.2f}")
        else:
            self.balance = config.INITIAL_CAPITAL
            print(f"Using paper trading balance: ${self.balance:.2f}")
            
        self.risk_manager = RiskManager(self.balance)
        self.regime_detector = MarketRegimeDetector()
        self.ml_predictor = MLPredictor()
        self.trade_history = []
        self.last_update = None
        self.notifier = TelegramNotifier()
        self._initialize_models()
        
        # Load trading mode
        self.paper_trading = config.PAPER_TRADING
        print(f"Trading Mode: {'Paper' if self.paper_trading else 'Live'}")
        print(f"Using {'Testnet' if config.USE_TESTNET else 'Live'} API")

    def _initialize_models(self):
        """Initialize ML model with some historical data"""
        print("Initializing models with historical data...")
        historical_data = self._fetch_initial_data()
        if historical_data is not None and not historical_data.empty:
            self.ml_predictor.train(historical_data)
        else:
            print("Warning: Could not initialize ML model with historical data")

    def _fetch_initial_data(self) -> pd.DataFrame:
        """Fetch initial historical data for model training"""
        try:
            # Calculate start date properly (e.g., 500 candles ago from now)
            start_date = (datetime.now() - timedelta(hours=config.ML_TRAINING_WINDOW)
                        ).strftime("%d %b %Y %H:%M:%S")
            
            # Fetch historical data
            df = data_fetcher.fetch_historical_data_chunked(
                self.client,
                symbol=config.SYMBOL,
                interval=config.INTERVAL,
                start_str=start_date
            )
            
            if df.empty:
                return pd.DataFrame()

            print(f"\nFetched {len(df)} candles for ML model training")
            return df
            
        except Exception as e:
            print(f"Error fetching initial data: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """Get current price without fetching full historical data"""
        try:
            # Get just the latest ticker price
            ticker = self.client.get_symbol_ticker(symbol=config.SYMBOL)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None

    def update_market_data(self):
        """Fetch and process latest market data"""
        try:
            # Fetch latest candle
            candles = self.client.get_klines(
                symbol=config.SYMBOL,
                interval=config.INTERVAL,
                limit=1
            )
            
            if not candles:
                return None
                
            current_candle = candles[0]
            current_price = float(current_candle[4])  # Close price
            
            # Update positions
            self._update_positions(current_price)
            
            # Get trading signal
            signal = self._generate_trading_signal()
            
            if signal == 1 and len(self.positions) < config.MAX_TRADES:
                self._open_position(current_price, 'long')
            
            self.last_update = datetime.now()
            
            return current_price
            
        except Exception as e:
            print(f"Error updating market data: {e}")
            return None

    def _generate_trading_signal(self) -> int:
        """Generate trading signal from combined strategy"""
        try:
            # Calculate start date based on ML training window
            num_hours = config.ML_TRAINING_WINDOW
            start_date = (datetime.now() - timedelta(hours=num_hours)).strftime("%d %b %Y %H:%M:%S")
            
            # Use chunked data fetching for proper historical context
            df = data_fetcher.fetch_historical_data_chunked(
                self.client,
                symbol=config.SYMBOL,
                interval=config.INTERVAL,
                start_str=start_date
            )
            
            if df.empty:
                return 0
                
            # Add indicators
            df = add_indicators(df)
            
            # Get regime
            df = self.regime_detector.detect_regime(df)
            current_regime = df['regime'].iloc[-1]
            
            # Get ML prediction
            ml_probability = self.ml_predictor.predict(df).iloc[-1]
            
            # Get standard strategy signal
            df = generate_signals(df)
            base_signal = df[config.COL_SIGNAL].iloc[-1]
            
            # Combine signals
            if (base_signal == 1 and  # Strategy signal is buy
                ml_probability > config.ML_CONFIDENCE_THRESHOLD and  # ML confirms with required confidence
                current_regime != 'downtrend' and  # Not in downtrend
                self.risk_manager.should_trade(current_regime)):  # Risk allows
                return 1
                
            return 0
            
        except Exception as e:
            print(f"Error generating trading signal: {e}")
            return 0

    def _update_positions(self, current_price: float):
        """Update all open positions"""
        for symbol, position in list(self.positions.items()):
            exit_signal = position.update(current_price)
            
            if exit_signal:
                self._close_position(position, exit_signal['price'], 
                                   exit_signal['reason'])

    def _open_position(self, current_price: float, side: str):
        """Open a new position"""
        try:
            # Calculate position size
            atr = self._calculate_atr()
            stop_loss = current_price - (atr * config.ATR_SL_MULTIPLIER)
            take_profit = current_price + (atr * config.ATR_TP_MULTIPLIER)
            
            risk_amount = self.balance * config.RISK_PCT_PER_TRADE
            position_size = self.risk_manager.get_position_size(
                current_price, stop_loss)
            
            if position_size * current_price < config.MIN_TRADE_USDT:
                print(f"Position size too small: ${position_size * current_price:.2f}")
                return

            if not self.paper_trading:
                # Execute real trade
                buy_order = self.trade_executor.place_market_buy(position_size)
                if not buy_order:
                    return
                
                # Place stop loss and take profit orders
                sl_order = self.trade_executor.place_stop_loss(position_size, stop_loss)
                tp_order = self.trade_executor.place_take_profit(position_size, take_profit)
                
                # Update actual entry price and quantity from the order
                entry_price = float(buy_order['fills'][0]['price'])
                actual_quantity = float(buy_order['executedQty'])
                
                position = Position(
                    symbol=config.SYMBOL,
                    entry_price=entry_price,
                    quantity=actual_quantity,
                    side=side,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                position.sl_order_id = sl_order['orderId'] if sl_order else None
                position.tp_order_id = tp_order['orderId'] if tp_order else None
                
            else:
                # Paper trading logic
                position = Position(
                    symbol=config.SYMBOL,
                    entry_price=current_price,
                    quantity=position_size,
                    side=side,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            # Store position
            self.positions[config.SYMBOL] = position
            
            # Send notification
            self.notifier.notify_trade_entry(
                symbol=config.SYMBOL,
                side=side,
                entry_price=position.entry_price,
                quantity=position.quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Log trade
            print(f"\nOpened {side} position:")
            print(f"Price: ${position.entry_price:.4f}")
            print(f"Size: {position.quantity:.4f}")
            print(f"Stop Loss: ${stop_loss:.4f}")
            print(f"Take Profit: ${take_profit:.4f}")
            
        except Exception as e:
            error_msg = f"Error opening position: {e}"
            print(error_msg)
            self.notifier.notify_error(error_msg)

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close an existing position"""
        try:
            if not self.paper_trading:
                # Cancel any existing SL/TP orders
                if hasattr(position, 'sl_order_id'):
                    self.trade_executor.cancel_order(position.sl_order_id)
                if hasattr(position, 'tp_order_id'):
                    self.trade_executor.cancel_order(position.tp_order_id)
                
                # Execute market sell
                sell_order = self.trade_executor.place_market_sell(position.quantity)
                if not sell_order:
                    return
                
                # Update actual exit price
                exit_price = float(sell_order['fills'][0]['price'])
            
            # Update position
            position.close(exit_price, reason)
            
            # Update balance
            self.balance = (self.trade_executor.get_balance() if not self.paper_trading 
                          else self.balance + position.pnl)
            
            # Remove from active positions
            del self.positions[position.symbol]
            
            # Store in history
            self.trade_history.append(position.to_dict())
            
            # Update risk manager
            self.risk_manager.record_trade(position.pnl)
            self.risk_manager.update_portfolio_value(self.balance)
            
            # Send notification
            self.notifier.notify_trade_exit(
                symbol=position.symbol,
                exit_price=exit_price,
                pnl=position.pnl,
                reason=reason
            )
            
            # Log trade
            print(f"\nClosed position - {reason}:")
            print(f"Entry: ${position.entry_price:.4f}")
            print(f"Exit: ${exit_price:.4f}")
            print(f"P/L: ${position.pnl:.2f}")
            print(f"New Balance: ${self.balance:.2f}")
            
        except Exception as e:
            error_msg = f"Error closing position: {e}"
            print(error_msg)
            self.notifier.notify_error(error_msg)

    def _calculate_atr(self) -> float:
        """Calculate current ATR value"""
        try:
            # Use last 100 periods for ATR calculation
            num_hours = 100  # Enough for ATR calculation
            start_date = (datetime.now() - timedelta(hours=num_hours)).strftime("%d %b %Y %H:%M:%S")
            
            df = data_fetcher.fetch_historical_data_chunked(
                self.client,
                symbol=config.SYMBOL,
                interval=config.INTERVAL,
                start_str=start_date
            )
            
            if df.empty:
                return 0.0
                
            df = add_indicators(df)
            return df[config.COL_ATR].iloc[-1]
            
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return 0.0

    def save_state(self):
        """Save current state to file"""
        state = {
            'balance': self.balance,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'trade_history': self.trade_history,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
        
        with open('paper_trader_state.json', 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load state from file"""
        try:
            with open('paper_trader_state.json', 'r') as f:
                state = json.load(f)
                
            self.balance = state['balance']
            self.trade_history = state['trade_history']
            self.last_update = datetime.fromisoformat(state['last_update']) \
                              if state['last_update'] else None
                              
            # Reconstruct positions
            self.positions = {}
            for symbol, pos_dict in state['positions'].items():
                position = Position(
                    symbol=pos_dict['symbol'],
                    entry_price=pos_dict['entry_price'],
                    quantity=pos_dict['quantity'],
                    side=pos_dict['side'],
                    stop_loss=pos_dict['stop_loss'],
                    take_profit=pos_dict['take_profit']
                )
                position.status = pos_dict['status']
                position.pnl = pos_dict['pnl']
                position.trailing_stop = pos_dict['trailing_stop']
                position.max_price = pos_dict['max_price']
                position.min_price = pos_dict['min_price']
                self.positions[symbol] = position
                
        except FileNotFoundError:
            print("No saved state found. Starting fresh.")