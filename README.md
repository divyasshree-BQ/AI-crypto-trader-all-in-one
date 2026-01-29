# AI Trading Bot on Uniswap - Complete Portfolio Management

An autonomous cryptocurrency trading bot that uses AI (GPT-4o or Claude) to make trading decisions on Uniswap (Base chain in this case) market & liquidity data.

Link to repo that uses CoW Intents to execute trades [https://github.com/divyasshree-BQ/AI-crypto-trader-all-in-one-CoW-Intent/tree/main](https://github.com/divyasshree-BQ/AI-crypto-trader-all-in-one-CoW-Intent/tree/main)

## Features

- AI-powered decision making using GPT-4o or Claude Sonnet
- Liquidity flow tracking for smart money detection
- Real-time slippage awareness and execution cost analysis
- Automated position management (buy, hold, sell)
- Portfolio tracking with PnL calculations

## Architecture

The bot operates in continuous cycles, performing the following steps:

1. Fetches trade data from the market
2. Analyzes liquidity events to detect smart money flows
3. Calculates slippage data for execution cost awareness
4. Sends market data to AI for analysis and decision making
5. Executes AI-recommended trades (buy/sell/hold)
6. Manages open positions and risk limits
7. Tracks portfolio performance and success metrics

## Safety Features

- Portfolio value limit: $10.00
- Maximum position size: $1.00
- Daily loss limit: $2.00
- Minimum confidence threshold: 10%
- Maximum open positions: 3
- Automatic position closure on stop-loss or target

## Requirements

- Python 3.8+
- Web3 wallet with private key
- Base mainnet RPC access (Infura or similar)
- BitQuery API key for market and liquidity data (create at https://account.bitquery.io/user/api_v2/access_tokens)
- OpenAI API key or Anthropic API key
- Initial USDC balance on Base chain

## Installation

1. Clone the repository
```bash
git clone https://github.com/divyasshree-BQ/AI-crypto-trader-all-in-one
cd AI-crypto-trader-all-in-one
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Get API Keys
- BitQuery: Create an API key at https://account.bitquery.io/user/api_v2/access_tokens
- OpenAI: Get your API key from https://platform.openai.com/api-keys
- Anthropic (optional): Get from https://console.anthropic.com/
- Infura: Get Base RPC endpoint from https://infura.io/

4. Configure environment variables
Copy `.env_example` to `.env` and fill in your values:
```
RPC_URL=https://base-mainnet.infura.io/v3/YOUR_INFURA_API_KEY
CHAIN_ID=8453
PRIVATE_KEY=your_wallet_private_key
ANTHROPIC_API_KEY=sk-ant-... (optional)
OPENAI_API_KEY=sk-proj-...
BITQUERY_API_KEY=ory_at_...
PORTFOLIO_SIZE_USD=10
MAX_POSITION_SIZE_USD=1
SLIPPAGE_TOLERANCE=1.0
GAS_LIMIT=300000
MAX_GAS_PRICE_GWEI=50
DAILY_LOSS_LIMIT_USD=2
MAX_OPEN_POSITIONS=3
MIN_CONFIDENCE_THRESHOLD=10
```

## Usage

Run the bot with:
```bash
python3 main.py
```

The bot will:
- Load environment variables and validate configuration
- Initialize wallet connection on Base mainnet (Chain ID: 8453)
- Display current balance and safety limits
- Start the trading loop with 60-second intervals

## Configuration

Key settings can be configured via environment variables:

- **RPC_URL**: Base mainnet RPC endpoint (Infura recommended)
- **CHAIN_ID**: Blockchain chain ID (8453 for Base)
- **BITQUERY_API_KEY**: Required for fetching trade and liquidity data
- **AI provider**: Supports OpenAI (GPT-4o) or Anthropic (Claude)
- **PORTFOLIO_SIZE_USD**: Maximum portfolio value in USD
- **MAX_POSITION_SIZE_USD**: Maximum size per position in USD
- **DAILY_LOSS_LIMIT_USD**: Maximum daily loss before stopping
- **MAX_OPEN_POSITIONS**: Maximum number of concurrent positions
- **MIN_CONFIDENCE_THRESHOLD**: Minimum AI confidence % to execute trades
- **SLIPPAGE_TOLERANCE**: Acceptable slippage percentage
- **GAS_LIMIT**: Maximum gas units per transaction
- **MAX_GAS_PRICE_GWEI**: Maximum gas price in Gwei

## Trading Logic

The AI analyzes multiple data sources:
- Recent trade volumes and price movements
- Liquidity addition/removal events
- Slippage estimates and execution costs
- Market maker activity
- Buy vs sell pressure

Based on this analysis, the AI generates trading actions with:
- Confidence level (0-100%)
- Reasoning for the decision
- Entry price and position size
- Target price for profit taking
- Stop loss price for risk management

## Performance Tracking

The bot tracks:
- Number of open positions
- Daily PnL (Profit and Loss)
- AI success rate
- Individual trade outcomes
- Gas costs per transaction

## Example Output

```
AI Trading Bot V2 - Enhanced with Liquidity Intelligence
============================================================
Features:
   • AI-powered decisions (GPT-4o)
   • Liquidity flow tracking (smart money)
   • Real slippage awareness (execution costs)
   • Enhanced risk management
============================================================
Wallet: ADDRESS_HERE
Chain ID: 8453
Balance: 0.003621 ETH
AI Provider: OpenAI (GPT-4o)
Safety Limits:
   - Portfolio: $10.0
   - Max Position: $1.0
   - Daily Loss Limit: $2.0
   - Min Confidence: 10%

Running trading loop with enhanced AI...
```

## Risk Disclaimer

This bot trades with real cryptocurrency on mainnet. Use at your own risk:
- Cryptocurrency trading carries significant risk
- Past performance does not guarantee future results
- Start with small amounts to test the bot
- Monitor the bot's performance regularly
- Be aware of gas costs on Base chain
- AI decisions may not always be profitable

## Technical Details

- **Blockchain**: Base (Chain ID: 8453)
- **DEX**: Uniswap V3
- **Router**: SwapRouter02 (0x2626664c2603336E57B271c5C0b26F421741e481)
- **Quote Contract**: QuoterV2 (0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a)
- **RPC**: https://mainnet.base.org

## License

MIT License

## Contributing

Contributions welcome. Please test thoroughly before submitting pull requests.

## Support

For issues or questions, please open an issue on GitHub.
