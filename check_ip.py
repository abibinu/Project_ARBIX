import requests
from binance.client import Client
import os
from dotenv import load_dotenv
import live_config as config
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def get_current_ip():
    """Get current server IP address"""
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except Exception as e:
        console.print(f"[bold red]Error getting IP:[/bold red] {e}")
        return None

def check_binance_ip_restrictions():
    """Check Binance API IP restrictions"""
    load_dotenv()
    
    # Get API keys
    api_key = os.getenv(config.LIVE_API_KEY_ENV)
    api_secret = os.getenv(config.LIVE_API_SECRET_ENV)
    
    if not api_key or not api_secret:
        console.print(Panel(
            "[bold red]Error:[/bold red] API keys not found in environment variables.\n"
            f"Please ensure {config.LIVE_API_KEY_ENV} and {config.LIVE_API_SECRET_ENV} are set.",
            title="Binance Key Status",
            border_style="red"
        ))
        return
    
    try:
        # Get current IP
        current_ip = get_current_ip()
        if not current_ip:
            return
            
        console.print(Panel(
            f"[bold cyan]🌐 Current IP Address:[/bold cyan] [yellow]{current_ip}[/yellow]",
            border_style="cyan"
        ))
        console.print("[bold yellow]Checking Binance API permissions...[/bold yellow]")
        
        # Initialize Binance client
        client = Client(api_key, api_secret)
        
        # Check API permissions
        try:
            api_permissions = client.get_api_key_permission()
            
            ip_restrict = api_permissions.get('ipRestrict', False)
            ip_list = api_permissions.get('ipList', [])
            
            perms_text = Text()
            perms_text.append("\n📋 API Permissions Status:\n\n", style="bold underline yellow")
            perms_text.append("IP Restriction Enabled: ", style="bold white")
            perms_text.append(f"{ip_restrict}\n", style="green" if ip_restrict else "yellow")
            perms_text.append("Allowed IPs: ", style="bold white")
            perms_text.append(f"{ip_list}\n", style="cyan")
            
            console.print(perms_text)
            
            if ip_restrict:
                if current_ip in ip_list:
                    console.print(Panel(
                        f"[bold green]✓ Success:[/bold green] Your current IP ({current_ip}) is whitelisted!",
                        border_style="green"
                    ))
                else:
                    warning_content = (
                        f"[bold red]⚠️ Warning:[/bold red] Your current IP ({current_ip}) is NOT whitelisted!\n\n"
                        "[bold white]To fix this:[/bold white]\n"
                        "1. Visit: https://www.binance.com/en/my/settings/api-management\n"
                        "2. Find your API key\n"
                        "3. Click 'Edit'\n"
                        "4. Add this IP address to the whitelist: [yellow]" + current_ip + "[/yellow]\n"
                        "5. Save changes"
                    )
                    console.print(Panel(warning_content, title="IP Mismatch Warning", border_style="yellow"))
            else:
                console.print(Panel(
                    "[bold yellow]⚠️ Note:[/bold yellow] IP restriction is not enabled for this key.\n"
                    "For security, it is highly recommended to enable IP restriction in Binance API settings.",
                    border_style="yellow"
                ))
                    
        except Exception as e:
            console.print(f"\n[bold red]Error checking API permissions:[/bold red] {e}")
            console.print(f"Your current IP is: [yellow]{current_ip}[/yellow]")
            console.print("Make sure to whitelist this IP in your Binance API settings")
            
    except Exception as e:
        console.print(f"[bold red]Error connecting to Binance:[/bold red] {e}")
        if 'current_ip' in locals() and current_ip:
            console.print(f"\nYour current IP is: [yellow]{current_ip}[/yellow]")
            console.print("Make sure to whitelist this IP in your Binance API settings")

if __name__ == "__main__":
    console.print(Panel("[bold green]Checking IP and Binance API restrictions...[/bold green]", border_style="green"))
    check_binance_ip_restrictions()