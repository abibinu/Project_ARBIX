import data_fetcher
from trade_executor import TradeExecutor
import live_config as config
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

def test_balance_fetch():
    console.print("[bold yellow]Initializing Binance client...[/bold yellow]")
    client = data_fetcher.get_binance_client()
    if not client:
        console.print("[bold red]Failed to initialize Binance client[/bold red]")
        return

    console.print(Panel(
        f"API Environment: [bold green]{'Testnet' if config.USE_TESTNET else 'Live'}[/bold green]",
        title="ARBIX Diagnostics",
        border_style="yellow"
    ))
    
    # Check API Permissions first
    console.print("\n[bold yellow]Checking API permissions...[/bold yellow]")
    try:
        perms = client.get_account()
        console.print("[green]✓ Can read account information[/green]")
        
        # Try to get open orders to check trading permission
        client.get_open_orders()
        console.print("[green]✓ Can access trading endpoints[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error checking permissions:[/bold red] {e}")
    
    console.print("\n[bold yellow]Checking account information & balances...[/bold yellow]")
    try:
        # Get account information including balances
        account_info = client.get_account()
        if 'balances' not in account_info:
            console.print("[bold red]Error: No balance information found in account response[/bold red]")
            return

        balances = [b for b in account_info['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        
        if not balances:
            console.print("[bold yellow]No non-zero balances found.[/bold yellow]")
        else:
            table = Table(title="Non-Zero Account Balances", show_header=True, header_style="bold magenta")
            table.add_column("Asset", style="cyan")
            table.add_column("Free Amount", justify="right", style="green")
            table.add_column("Locked Amount", justify="right", style="red")
            table.add_column("Est. Value (USDT)", justify="right", style="bold yellow")
            
            # Calculate total USDT value
            total_usdt = 0
            for balance in balances:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                # For USDT, add directly
                if asset == 'USDT':
                    asset_value = free + locked
                    total_usdt += asset_value
                    table.add_row(
                        asset, 
                        f"{free:.8f}", 
                        f"{locked:.8f}", 
                        f"${asset_value:.2f}"
                    )
                else:
                    try:
                        # Get current price for non-USDT assets
                        ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        asset_value = (free + locked) * price
                        total_usdt += asset_value
                        table.add_row(
                            asset, 
                            f"{free:.8f}", 
                            f"{locked:.8f}", 
                            f"${asset_value:.2f}"
                        )
                    except:
                        table.add_row(
                            asset, 
                            f"{free:.8f}", 
                            f"{locked:.8f}", 
                            "N/A"
                        )
            
            console.print(table)
            console.print(Panel(
                f"[bold cyan]Total Portfolio Value:[/bold cyan] [bold yellow]${total_usdt:.4f} USDT[/bold yellow]",
                border_style="cyan"
            ))
            
    except Exception as e:
        console.print(f"[bold red]Error checking account information:[/bold red] {e}")
        
    console.print("\n[bold yellow]Checking trading status...[/bold yellow]")
    try:
        trading_status = client.get_account_status()
        console.print(f"Account status: [cyan]{trading_status}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Error getting trading status:[/bold red] {e}")

    console.print("\n[bold yellow]Checking account API trading status...[/bold yellow]")
    try:
        api_trading_status = client.get_account_api_trading_status()
        console.print(f"API trading status: [cyan]{api_trading_status}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Error getting API trading status:[/bold red] {e}")

if __name__ == "__main__":
    test_balance_fetch()