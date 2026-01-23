"""
Market Data Fetcher - Fetches live data from Bitquery for the real trading bot
"""
import os
import requests
from typing import Dict, List
from collections import defaultdict
import json


BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY')


def fetch_base_dex_data(limit: int = 100, required_tokens: List[Dict] = None) -> Dict:
    """
    Fetch recent DEX trades from Base network via Bitquery REST API
    """
    url = "https://streaming.bitquery.io/graphql"

    query = """
    query BaseDEXTrades {
        EVM(network: base) {
            DEXTradeByTokens(
                limit: {count: 200}
                orderBy: {descending: Block_Time}
                where: {
                    TransactionStatus: {Success: true}
                    Trade: {
                        Side: {
                            Currency: {
                                SmartContract: {is: "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"}
                            }
                        }
                    }
                }
            ) {
                Block {
                    Time
                }
                Trade {
                    Dex {
                        ProtocolName
                    }
                    PriceInUSD
                    Side {
                        Type
                        AmountInUSD
                        Currency {
                            Symbol
                            SmartContract
                        }
                    }
                    Currency {
                        Symbol
                        SmartContract
                    }
                }
            }
        }
    }
    """

    try:
        response = requests.post(
            url,
            json={
                'query': query
            },
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {BITQUERY_API_KEY}'
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ Bitquery API error: {response.status_code}")
            return None

        data = response.json()
        trades = data.get('data', {}).get('EVM', {}).get('DEXTradeByTokens', [])

        if not trades:
            print("⚠️  No trades returned from Bitquery")
            return None

        return process_trades(trades, required_tokens)

    except Exception as e:
        print(f"❌ Error fetching Bitquery data: {e}")
        return None


def process_trades(trades: List[Dict], required_tokens: List[Dict] = None) -> Dict:
    """
    Process raw trades into market summary format
    required_tokens: List of dicts with 'symbol' and/or 'contract_address' to ensure inclusion
    """
    market_map = defaultdict(lambda: {
        'trades': [],
        'total_volume': 0,
        'buy_volume': 0,
        'sell_volume': 0
    })

    for trade in trades:
        try:
            token = trade['Trade']['Currency']
            symbol = token.get('Symbol') or token.get('SmartContract', '')[:10]

            price = float(trade['Trade'].get('PriceInUSD', 0) or 0)
            volume_str = trade['Trade']['Side'].get('AmountInUSD', '0')
            volume = float(volume_str) if volume_str else 0
            side = trade['Trade']['Side']['Type']
            contract = token.get('SmartContract', '')

            if volume == 0 or price == 0 or not symbol:
                continue

            market_map[symbol]['trades'].append({
                'time': trade['Block']['Time'],
                'price': price,
                'volume': volume,
                'side': side
            })
            market_map[symbol]['total_volume'] += volume
            market_map[symbol]['contract_address'] = contract

            if side == 'buy':
                market_map[symbol]['buy_volume'] += volume
            else:
                market_map[symbol]['sell_volume'] += volume

        except Exception as e:
            continue

    # Build list of all markets with RAW volumes (let AI calculate ratios)
    all_markets = [
        {
            'symbol': symbol,
            'volume': data['total_volume'],
            'buy_volume': data['buy_volume'],
            'sell_volume': data['sell_volume'],
            'trade_count': len(data['trades']),
            'recent_price': data['trades'][-1]['price'] if data['trades'] else 0,
            'contract_address': data.get('contract_address', '')
        }
        for symbol, data in market_map.items()
    ]

    # Return ALL markets (let AI decide what's relevant, not us)
    # Sort by volume for convenience only
    all_markets_sorted = sorted(all_markets, key=lambda x: x['volume'], reverse=True)

    total_volume = sum(m['volume'] for m in all_markets_sorted)

    return {
        'top_markets': all_markets_sorted,
        'total_volume': total_volume,
        'market_count': len(all_markets_sorted),
        'timestamp': trades[0]['Block']['Time'] if trades else None
    }


def get_market_data_for_trading(required_tokens: List[Dict] = None) -> Dict:
    """
    Get formatted market data ready for Claude analysis
    required_tokens: List of dicts with 'symbol' and/or 'contract_address' to ensure inclusion
    """
    data = fetch_base_dex_data(limit=200, required_tokens=required_tokens)

    if not data:
        return None

    print(f"📊 Fetched {len(data['top_markets'])} markets, ${data['total_volume']:.2f} volume")

    # Print top markets (raw data only)
    for i, market in enumerate(data['top_markets'][:5], 1):
        print(f"   {i}. {market['symbol']}: ${market['volume']:.2f} vol, {market['trade_count']} trades")

    return data


if __name__ == "__main__":
    print("🔍 Testing Market Data Fetcher...")
    data = get_market_data_for_trading()

    if data:
        print(f"\n✅ Successfully fetched market data")
        print(json.dumps(data, indent=2))
    else:
        print("\n❌ Failed to fetch market data")
