# SCALPARS - Automated Crypto Futures Trading Platform

An automated trading bot for cryptocurrency futures on Binance, featuring technical analysis indicators (EMA, RSI, ADX), configurable trading strategies, and both paper trading and live trading modes.

## Features

- **Real-time Market Data**: Top 50 futures pairs by volume with live indicators
- **Technical Indicators**: EMA (5, 8, 13, 20), RSI (12), ADX
- **Signal Generation**: Automated LONG/SHORT signals based on EMA stacking and momentum
- **Confidence Levels**: LOW, MEDIUM, HIGH, EXTREME with configurable leverage
- **Risk Management**: Stop Loss, Take Profit, and Trailing Stop functionality
- **Paper Trading**: Test strategies with $10,000 virtual balance
- **Live Trading**: Execute real trades on Binance Futures
- **Performance Tracking**: Comprehensive metrics for closed orders

## Quick Start

### Option 1: Using the run script (Recommended)

```bash
cd "/Users/guillearslanian/Downloads/NOFA AI"
python3 run.py
```

This will:
1. Create a virtual environment (if not exists)
2. Install all dependencies
3. Start the server on http://localhost:8000

### Option 2: Manual setup

```bash
# Navigate to the project directory
cd "/Users/guillearslanian/Downloads/NOFA AI"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in your browser.

## Configuration

### Environment Variables

Edit the `.env` file to configure:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret_key
```

### Trading Configuration

All trading parameters can be configured through the UI:

- **Investment Settings**: Fixed amount or percentage of balance
- **Safe Reserve**: Amount to keep uninvested
- **Signal Thresholds**: RSI and ADX levels for each confidence tier
- **Leverage**: Min/Max leverage per confidence level
- **Risk Management**: Stop Loss, Take Profit, Pullback triggers

## Trading Logic

### Signal Generation

**LONG Signal** (Bullish EMA Stack: EMA5 > EMA8 > EMA13 > EMA20):
- EXTREME: RSI < 30, ADX > 35, Volume > 1.5x average
- HIGH: RSI < 30, ADX > 25
- MEDIUM: RSI < 35, ADX > 20
- LOW: RSI < 55

**SHORT Signal** (Bearish EMA Stack: EMA5 < EMA8 < EMA13 < EMA20):
- EXTREME: RSI > 70, ADX > 35, Volume > 1.5x average
- HIGH: RSI > 70, ADX > 25
- MEDIUM: RSI > 65, ADX > 20
- LOW: RSI > 45

### Position Management

1. **Entry**: Opens position when signal conditions are met
2. **Stop Loss**: Closes position if P&L drops below threshold
3. **Take Profit Minimum**: Position becomes eligible for trailing stop
4. **Trailing Stop**: Closes position if P&L drops from peak by pullback amount

## UI Components

1. **Header**: Bot status, timer, start/pause controls, paper/live toggle
2. **Balance Cards**: USDT, BNB, Open Orders, Total Portfolio
3. **Performance Metrics**: Win rates, P&L, durations, investment totals
4. **Pairs Table**: Top 50 pairs with all indicators and signals
5. **Orders Tabs**: Transactions, Open Orders, Closed Orders
6. **Configuration Panel**: All adjustable trading parameters

## API Endpoints

- `GET /api/status` - Bot status and runtime
- `POST /api/start` - Start trading
- `POST /api/pause` - Pause trading (still closes positions)
- `GET /api/balance` - Account balances
- `GET /api/pairs` - Top 50 pairs with indicators
- `GET /api/orders/open` - Open positions
- `GET /api/orders/closed` - Closed positions
- `GET /api/performance` - Performance metrics
- `GET /api/config` - Trading configuration
- `PUT /api/config` - Update configuration

## Important Notes

- **Paper Trading First**: Always test with paper trading before using live mode
- **Risk Management**: Start with conservative settings
- **API Keys**: Keep your API keys secure, never commit them
- **Fees**: Default fee is 0.04% (Binance Futures taker fee)

## File Structure

```
NOFA AI/
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── database.py          # Database setup
├── models.py            # SQLAlchemy models
├── services/
│   ├── binance_service.py    # Binance API wrapper
│   ├── indicators.py         # Technical indicators
│   └── trading_engine.py     # Trading logic
├── templates/
│   └── index.html       # Frontend UI
├── static/              # Static files
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── run.py              # Easy start script
```

## License

For personal use only. Trade at your own risk.
