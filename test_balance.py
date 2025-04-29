import data_fetcher
from trade_executor import TradeExecutor
import live_config as config

def test_balance_fetch():
    print("Initializing Binance client...")
    client = data_fetcher.get_binance_client()
    if not client:
        print("Failed to initialize Binance client")
        return

    print(f"\nTesting on {'Testnet' if config.USE_TESTNET else 'Live'} API")
    
    # Check API Permissions first
    print("\nChecking API permissions...")
    try:
        perms = client.get_account()
        print("API Permissions:")
        print("✓ Can read account information")
        
        # Try to get open orders to check trading permission
        client.get_open_orders()
        print("✓ Can access trading endpoints")
        
    except Exception as e:
        print(f"Error checking permissions: {e}")
    
    print("\nChecking account information...")
    try:
        # Get account information including balances
        account_info = client.get_account()
        if 'balances' not in account_info:
            print("Error: No balance information found in account response")
            return

        print("\nAccount Information:")
        balances = [b for b in account_info['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        
        if not balances:
            print("No non-zero balances found")
        else:
            # Calculate total USDT value
            total_usdt = 0
            for balance in balances:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                # For USDT, add directly
                if asset == 'USDT':
                    total_usdt += free + locked
                    print(f"{asset:<8} Free: {free:>10.8f}, Locked: {locked:>10.8f}")
                else:
                    try:
                        # Get current price for non-USDT assets
                        ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        asset_value = (free + locked) * price
                        total_usdt += asset_value
                        print(f"{asset:<8} Free: {free:>10.8f}, Locked: {locked:>10.8f} (≈ ${asset_value:.2f})")
                    except:
                        print(f"{asset:<8} Free: {free:>10.8f}, Locked: {locked:>10.8f}")
            
            print(f"\nTotal Portfolio Value: ${total_usdt:.2f} USDT")
    except Exception as e:
        print(f"Error checking account information: {e}")
        
    print("\nTrying to get trading status...")
    try:
        trading_status = client.get_account_status()
        print("Account status:", trading_status)
    except Exception as e:
        print(f"Error getting trading status: {e}")

    print("\nChecking account API trading status...")
    try:
        api_trading_status = client.get_account_api_trading_status()
        print("API trading status:", api_trading_status)
    except Exception as e:
        print(f"Error getting API trading status: {e}")

if __name__ == "__main__":
    test_balance_fetch()