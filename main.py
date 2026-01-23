#!/usr/bin/env python3
"""
V2 Trading Bot Launcher - Enhanced AI with Liquidity Intelligence
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Manually load .env file
env_path = Path('.env')
if env_path.exists():
    for line in env_path.read_text().split('\n'):
        line = line.strip()
        if line and '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
    print("Environment variables loaded")
else:
    print(".env file not found!")
    sys.exit(1)

# Verify critical variables
required_vars = ['RPC_URL', 'PRIVATE_KEY']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# Check for at least one AI API key
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')
if not anthropic_key and not openai_key:
    print("Missing AI API key: At least one of ANTHROPIC_API_KEY or OPENAI_API_KEY must be set")
    sys.exit(1)

print(f"All required variables present")
if openai_key:
    print(f"   OpenAI API key found (primary)")
if anthropic_key:
    print(f"   Anthropic API key found (disabled - no credits)")
print()

# Import and run bot
from trading_bot import AITradingBotV2

if __name__ == "__main__":
    print("🤖 AI Trading Bot V2 - Enhanced with Liquidity Intelligence")
    print("=" * 60)
    print("✨ Features:")
    print("   • AI-powered decisions (GPT-4o)")
    print("   • Liquidity flow tracking (smart money)")
    print("   • Real slippage awareness (execution costs)")
    print("   • Enhanced risk management")
    print("=" * 60)

    try:
        bot = AITradingBotV2()
        print("\n🚀 Bot initialized successfully!")
        print(f"💰 Starting balance: {bot._get_balance():.6f} ETH")
        print("\n⏰ Running trading loop with enhanced AI...")
        print("=" * 60)
        bot.run(interval=60)
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
