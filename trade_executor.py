"""
Handles actual trading operations for both testnet and live
"""
from binance.exceptions import BinanceAPIException
from decimal import Decimal, ROUND_DOWN
import time
import live_config as config

class TradeExecutor:
    def __init__(self, client, notifier=None):
        self.client = client
        self.notifier = notifier
        self._load_exchange_info()

    def _load_exchange_info(self):
        """Load symbol info and trading rules"""
        try:
            exchange_info = self.client.get_exchange_info()
            self.symbol_info = next(
                (s for s in exchange_info['symbols'] if s['symbol'] == config.SYMBOL),
                None
            )
            if not self.symbol_info:
                raise ValueError(f"Symbol {config.SYMBOL} not found in exchange info")

            # Extract precision info
            self.price_precision = self._get_precision('price')
            self.qty_precision = self._get_precision('qty')

        except Exception as e:
            error_msg = f"Error loading exchange info: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)

    def _get_precision(self, filter_type):
        """Get precision for price or quantity"""
        if not self.symbol_info:
            return 8  # Default precision

        if filter_type == 'price':
            price_filter = next(
                (f for f in self.symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'),
                None
            )
            if price_filter:
                tick_size = Decimal(price_filter['tickSize'])
                return abs(tick_size.as_tuple().exponent)
        elif filter_type == 'qty':
            lot_filter = next(
                (f for f in self.symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
                None
            )
            if lot_filter:
                step_size = Decimal(lot_filter['stepSize'])
                return abs(step_size.as_tuple().exponent)
        
        return 8  # Default precision

    def format_number(self, number, precision):
        """Format number to required precision"""
        return f"{{:.{precision}f}}".format(float(number))

    def place_market_buy(self, quantity):
        """Place a market buy order"""
        try:
            quantity = self.format_number(quantity, self.qty_precision)
            order = self.client.order_market_buy(
                symbol=config.SYMBOL,
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            error_msg = f"Binance API error placing market buy: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return None

    def place_market_sell(self, quantity):
        """Place a market sell order"""
        try:
            quantity = self.format_number(quantity, self.qty_precision)
            order = self.client.order_market_sell(
                symbol=config.SYMBOL,
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            error_msg = f"Binance API error placing market sell: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return None

    def place_stop_loss(self, quantity, stop_price):
        """Place a stop-loss order"""
        try:
            quantity = self.format_number(quantity, self.qty_precision)
            stop_price = self.format_number(stop_price, self.price_precision)
            order = self.client.create_order(
                symbol=config.SYMBOL,
                side='SELL',
                type='STOP_LOSS_LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                stopPrice=stop_price,
                price=stop_price  # Limit price same as stop price
            )
            return order
        except BinanceAPIException as e:
            error_msg = f"Binance API error placing stop loss: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return None

    def place_take_profit(self, quantity, price):
        """Place a take-profit limit order"""
        try:
            quantity = self.format_number(quantity, self.qty_precision)
            price = self.format_number(price, self.price_precision)
            order = self.client.create_order(
                symbol=config.SYMBOL,
                side='SELL',
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )
            return order
        except BinanceAPIException as e:
            error_msg = f"Binance API error placing take profit: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return None

    def cancel_order(self, order_id):
        """Cancel an existing order"""
        try:
            return self.client.cancel_order(
                symbol=config.SYMBOL,
                orderId=order_id
            )
        except BinanceAPIException as e:
            error_msg = f"Binance API error canceling order: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return None

    def get_all_orders(self):
        """Get all orders for the symbol"""
        try:
            return self.client.get_all_orders(symbol=config.SYMBOL)
        except BinanceAPIException as e:
            error_msg = f"Binance API error getting orders: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return []

    def get_open_orders(self):
        """Get all open orders for the symbol"""
        try:
            return self.client.get_open_orders(symbol=config.SYMBOL)
        except BinanceAPIException as e:
            error_msg = f"Binance API error getting open orders: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return []

    def get_position(self):
        """Get current position information"""
        try:
            account = self.client.get_account()
            balance = next(
                (b for b in account['balances'] 
                 if b['asset'] == config.SYMBOL.replace('USDT', '')),
                None
            )
            return float(balance['free']) if balance else 0.0
        except BinanceAPIException as e:
            error_msg = f"Binance API error getting position: {e}"
            print(error_msg)
            if self.notifier:
                self.notifier.notify_error(error_msg)
            return 0.0

    def get_balance(self, asset='USDT'):
        """Get balance for specific asset with retry mechanism"""
        max_retries = 3
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get account info
                account = self.client.get_account()
                
                # Find the specific asset balance
                for balance in account['balances']:
                    if balance['asset'] == asset:
                        free_balance = float(balance['free'])
                        locked_balance = float(balance['locked'])
                        total_balance = free_balance + locked_balance
                        
                        if total_balance > 0:
                            print(f"Current {asset} balance: Free=${free_balance:.2f}, Locked=${locked_balance:.2f}, Total=${total_balance:.2f}")
                            return total_balance
                        else:
                            print(f"Attempt {attempt + 1}/{max_retries}: Balance appears to be zero, waiting {retry_delay}s for sync...")
                            time.sleep(retry_delay)
                            continue
                
                print(f"No {asset} balance found")
                time.sleep(retry_delay)
                
            except Exception as e:
                error_msg = f"Binance API error getting balance (attempt {attempt + 1}): {e}"
                print(error_msg)
                if attempt < max_retries - 1:  # Don't notify on retries
                    time.sleep(retry_delay)
                else:  # Notify only on final attempt
                    if self.notifier:
                        self.notifier.notify_error(error_msg)
                    return 0.0
        
        return 0.0