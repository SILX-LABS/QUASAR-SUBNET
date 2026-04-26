"""Market data endpoints: price, TMC config, metagraph."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import NETUID, TMC_BASE
from external import get_metagraph as load_metagraph, get_price as load_price

router = APIRouter()


@router.get("/api/metagraph", tags=["Metagraph"], summary="Full subnet metagraph",
         description="""Returns all 256 UIDs with on-chain data: hotkey, coldkey, stake, trust, consensus, incentive, emission, and dividends.

**Cached for 60s** - background refreshes keep data fresh without blocking requests.

Response includes:
- `block`: Current Bittensor block number
- `n`: Number of UIDs in the subnet (256)
- `neurons[]`: Array of all UIDs with their on-chain metrics
""",
         response_description="Metagraph with all 256 UIDs and their on-chain metrics")
def get_metagraph():
    return JSONResponse(content=load_metagraph(), headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})


@router.get("/api/price", tags=["Market"], summary="Token price and market data",
         description="""Returns SN24 alpha token pricing, TAO/USD rate, pool liquidity, emission, and volume.

Response includes:
- `alpha_price_tao` / `alpha_price_usd`: Current alpha token price
- `tao_usd`: TAO/USD exchange rate (via CoinGecko)
- `alpha_in_pool` / `tao_in_pool`: DEX pool liquidity
- `marketcap_tao`: Total market cap in TAO
- `emission_pct`: Current emission allocation percentage
- `price_change_1h`, `_24h`, `_7d`: Price change percentages
- `miners_tao_per_day`: Total TAO earned by miners per day

**Cached for 30s.**
""")
def get_price():
    return JSONResponse(content=load_price(), headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})


@router.get("/api/tmc-config", tags=["Market"], summary="TaoMarketCap SSE config",
         description="Returns SSE (Server-Sent Events) URLs for real-time price and subnet data from TaoMarketCap. Used by the dashboard for live price updates.")
def get_tmc_config():
    return {
        "sse_price_url": f"{TMC_BASE}/public/v1/sse/subnets/prices/",
        "sse_subnet_url": f"{TMC_BASE}/public/v1/sse/subnets/{NETUID}/",
        "netuid": NETUID,
    }
