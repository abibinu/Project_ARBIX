import requests
from binance.client import Client
import os
from dotenv import load_dotenv
import live_config as config

def get_current_ip():
    """Get current server IP address"""
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except Exception as e:
        print(f"Error getting IP: {e}")
        return None

def check_binance_ip_restrictions():
    """Check Binance API IP restrictions"""
    load_dotenv()
    
    # Get API keys
    api_key = os.getenv(config.LIVE_API_KEY_ENV)
    api_secret = os.getenv(config.LIVE_API_SECRET_ENV)
    
    if not api_key or not api_secret:
        print("Error: API keys not found in environment variables")
        return
    
    try:
        # Get current IP
        current_ip = get_current_ip()
        if not current_ip:
            return
            
        print(f"\n🌐 Current IP Address: {current_ip}")
        print("\nChecking Binance API permissions...")
        
        # Initialize Binance client
        client = Client(api_key, api_secret)
        
        # Check API permissions
        try:
            api_permissions = client.get_api_key_permission()
            
            print("\n📋 API Permissions Status:")
            print(f"IP Restriction Enabled: {api_permissions.get('ipRestrict', False)}")
            print(f"Allowed IPs: {api_permissions.get('ipList', [])}")
            
            if api_permissions.get('ipRestrict', False):
                if current_ip in api_permissions.get('ipList', []):
                    print(f"\n✅ Success: Your current IP ({current_ip}) is whitelisted!")
                else:
                    print(f"\n⚠️ Warning: Your current IP ({current_ip}) is NOT whitelisted!")
                    print("\nTo fix this:")
                    print("1. Visit: https://www.binance.com/en/my/settings/api-management")
                    print("2. Find your API key")
                    print("3. Click 'Edit'")
                    print(f"4. Add this IP address: {current_ip}")
                    print("5. Save changes")
                    
        except Exception as e:
            print(f"\nError checking API permissions: {e}")
            print(f"\nYour current IP is: {current_ip}")
            print("Make sure to whitelist this IP in your Binance API settings")
            
    except Exception as e:
        print(f"Error connecting to Binance: {e}")
        if current_ip:
            print(f"\nYour current IP is: {current_ip}")
            print("Make sure to whitelist this IP in your Binance API settings")

if __name__ == "__main__":
    print("Checking IP and Binance API restrictions...")
    check_binance_ip_restrictions()