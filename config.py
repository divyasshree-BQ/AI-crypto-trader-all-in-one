"""
DEX Configuration for Base Chain
Uniswap V3 Router and common token addresses
"""

# Base Mainnet (Chain ID: 8453)
BASE_MAINNET = {
    "chain_id": 8453,
    "rpc_url": "https://mainnet.base.org",

    # Uniswap V3 on Base
    "router_v3": "0x2626664c2603336E57B271c5C0b26F421741e481",  # SwapRouter02
    "quoter_v2": "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
    "factory": "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",

    # Common tokens on Base
    "tokens": {
        "WETH": "0x4200000000000000000000000000000000000006",  # Wrapped ETH
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Native USDC
        "USDbC": "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA", # Bridged USDC
        "DAI": "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
        "cbBTC": "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",  # Coinbase Wrapped BTC
        "USDT": "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",
    }
}

# Uniswap V3 SwapRouter02 ABI (minimal)
SWAPROUTER_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "tokenIn", "type": "address"},
                    {"internalType": "address", "name": "tokenOut", "type": "address"},
                    {"internalType": "uint24", "name": "fee", "type": "uint24"},
                    {"internalType": "address", "name": "recipient", "type": "address"},
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
                    {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
                ],
                "internalType": "struct IV3SwapRouter.ExactInputSingleParams",
                "name": "params",
                "type": "tuple"
            }
        ],
        "name": "exactInputSingle",
        "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function"
    }
]

# ERC20 ABI (minimal)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]

# Fee tiers for Uniswap V3 (in basis points)
FEE_TIERS = {
    "LOW": 500,      # 0.05%
    "MEDIUM": 3000,  # 0.3%
    "HIGH": 10000    # 1%
}
