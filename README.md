# Project Arbix: Algorithmic Crypto Trading System

Project Arbix is an automated, quantitative cryptocurrency trading system that integrates technical analysis indicators, machine learning model confirmation, and adaptive market regime detection. It is designed to execute backtests, manage simulated paper trading, and perform live trading execution on cryptocurrency markets via the Binance API.

---

## Features

- **Technical Analysis Engine**: Automatically computes short and long Exponential Moving Averages (EMA), Relative Strength Index (RSI), and Average True Range (ATR).
- **Machine Learning Signal Filtering**: Utilizes a Random Forest Classifier trained on technical features to predict high-probability entries and filter out low-confidence trade signals.
- **Market Regime Detection**: Classifies the current market state into uptrend, downtrend, volatile, ranging, or normal, dynamically adjusting strategy parameters (stop losses, take profits, entry thresholds) based on market structure.
- **Advanced Risk Management**: Employs risk-based position sizing, ATR-based Stop Loss (SL) and Take Profit (TP) calculations, trailing stop adjustments, and portfolio drawdown controls.
- **Real-Time Terminal Dashboard**: Renders a styled terminal interface using rich panels, tables, and logging widgets to monitor bot status, market prices, portfolio balance, active positions, and trade history in real-time.
- **TradingView-Inspired Visualizer**: Generates high-fidelity dark-themed charts plotting asset price action, trade execution entry/exits, RSI levels, equity valuation curves, and shaded backgrounds representing detected market regimes.

---

## Project Structure

- `main.py`: Entry point for backtesting simulation, orchestration, and matplotlib analytics generation.
- `backtester.py`: Core simulation engine simulating asset holding, transaction fee calculation, and stop-loss/take-profit exit criteria.
- `indicators.py`: Calculations for RSI, ATR, and short/long/long-term EMAs.
- `strategy.py`: Generates the baseline crossover and RSI trading signals.
- `market_regime.py`: Analyzes price regression, volatility quantiles, and trend strength to classify market state and fetch optimal parameters.
- `ml_predictor.py`: Configures features, labels, splits, scales, trains, and executes the Random Forest binary classifier.
- `risk_manager.py`: Manages trade limits, position sizing based on risk percentage, and drawdown limitations.
- `paper_trader.py`: The trading implementation layer managing paper/live portfolios, order execution, position tracking, and logging.
- `trade_executor.py`: Interface for connecting to the Binance API to query balances and place market or OCO orders.
- `reporting.py`: Computes backtesting performance statistics and draws high-fidelity matplotlib figures and live terminal dashboards.
- `notifications.py`: Handles Telegram alert integration for real-time trade execution notices.
- `config.py`: Core static parameters for historical testing.
- `live_config.py`: Live and paper trading operational parameters.
- `check_ip.py`: Network diagnostic utility verifying external IP whitelisting settings in Binance API keys.
- `test_balance.py`: Query tool checking API permissions, active non-zero balances, and current total portfolio value.

---

## Installation & Setup

1. **Clone the Repository**:
   Clone this repository to your local directory.

2. **Install Dependencies**:
   Install the required Python libraries using pip:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory of the project and populate it with your Binance API keys and Telegram bot parameters:
   ```env
   BINANCE_LIVE_API_KEY=your_live_api_key_here
   BINANCE_LIVE_API_SECRET=your_live_api_secret_here
   BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
   BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here
   ```

4. **Adjust Trading Parameters**:
   Review parameters inside `config.py` and `live_config.py` to customize symbols, intervals, risk allocation, target multipliers, and notification channels.

---

## Usage

### 1. Run Diagnostics
Verify connection capabilities, API keys, and IP whitelist status before execution:
```bash
python check_ip.py
python test_balance.py
```

### 2. Run Backtest Simulation
Run a historical simulation on past candle data to evaluate performance:
```bash
python main.py
```
This prints a structured terminal performance summary and generates analytical graphs.

### 3. Run Paper Trading
Launch the paper trading agent in simulated mode using real-time price feeds:
- Verify `PAPER_TRADING = True` in `live_config.py`.
- Start execution:
  ```bash
  python run_paper_trading.py
  ```

### 4. Run Live Trading
Launch the live trading agent to execute real orders:
- Verify `PAPER_TRADING = False` in `live_config.py`.
- Start execution:
  ```bash
  python run_live_trading.py
  ```
