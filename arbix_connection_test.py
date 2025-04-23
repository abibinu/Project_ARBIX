import os
from binance.client import Client
import pandas as pd # Good habit to import pandas early on

# --- Configuration ---
# Load API keys from environment variables
api_key = os.getenv("BINANCE_TESTNET_API_KEY")
api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

# Check if keys were loaded
if not api_key or not api_secret:
    print("Error: Binance API keys not found.")
    print("Please set the BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET environment variables.")
    exit() # Stop the script if keys are missing

# --- Initialize Binance Client ---
# Crucially, set testnet=True for the Test Network!
client = Client(api_key, api_secret, testnet=True)
print("Successfully connected to Binance Testnet Client.")

# --- Test API Call 1: Get Account Information ---
try:
    print("\nAttempting to fetch account information...")
    account_info = client.get_account()

    # Convert the 'balances' list into a Pandas DataFrame for better readability
    balances = pd.DataFrame(account_info['balances'])

    # Convert 'free' and 'locked' columns to numeric (they come as strings)
    balances['free'] = pd.to_numeric(balances['free'])
    balances['locked'] = pd.to_numeric(balances['locked'])

    # Filter out assets with zero balance
    balances = balances[(balances['free'] > 0) | (balances['locked'] > 0)]

    print("\n--- Testnet Account Balances ---")
    print(balances)
    print("-" * 30) # Separator

except Exception as e:
    print(f"\nError fetching account information: {e}")
    print("Check API key permissions and network connection.")

# --- Test API Call 2: Get Ticker Price for a Specific Pair ---
try:
    symbol = 'BTCUSDT' # A common pair, should exist on Testnet
    print(f"\nAttempting to fetch ticker price for {symbol}...")
    ticker = client.get_symbol_ticker(symbol=symbol)

    print(f"\n--- Current {symbol} Ticker ---")
    print(ticker)
    print("-" * 30) # Separator

except Exception as e:
    print(f"\nError fetching ticker price for {symbol}: {e}")

print("\nConnection test script finished.")