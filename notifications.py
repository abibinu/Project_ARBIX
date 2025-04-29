"""
Handles notifications for trading events using Telegram
"""
import os
import requests
import live_config as config

class TelegramNotifier:
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = config.ENABLE_NOTIFICATIONS and self.bot_token and self.chat_id
        
        if self.enabled:
            print("Telegram notification sent!")
        else:
            print("Telegram notifications disabled - check bot token and chat ID")

    def send_message(self, message: str) -> bool:
        """Send a message via Telegram"""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
            return False

    def notify_trade_entry(self, symbol: str, side: str, entry_price: float, 
                          quantity: float, stop_loss: float, take_profit: float):
        """Send trade entry notification"""
        message = (
            f"🎯 <b>New Trade Entry</b>\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
            f"Entry Price: ${entry_price:.4f}\n"
            f"Quantity: {quantity:.4f}\n"
            f"Stop Loss: ${stop_loss:.4f}\n"
            f"Take Profit: ${take_profit:.4f}"
        )
        self.send_message(message)

    def notify_trade_exit(self, symbol: str, exit_price: float, pnl: float, 
                         reason: str):
        """Send trade exit notification"""
        emoji = "🟢" if pnl > 0 else "🔴"
        message = (
            f"{emoji} <b>Trade Exit</b>\n"
            f"Symbol: {symbol}\n"
            f"Exit Price: ${exit_price:.4f}\n"
            f"P/L: ${pnl:.2f}\n"
            f"Reason: {reason}"
        )
        self.send_message(message)

    def notify_error(self, error_message: str):
        """Send error notification"""
        message = f"⚠️ <b>Error</b>\n{error_message}"
        self.send_message(message)