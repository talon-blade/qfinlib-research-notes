"""Example pricing a vanilla swaption using generated market data."""

import os
import sys
from datetime import timedelta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qfinlib.instruments.rates.option.swaption import Swaption
from qfinlib.instruments.rates.swap.irs import Swap
from qfinlib.market.data import DataLoader, RandomMarketDataProvider
from qfinlib.pricing.pricers.rates.swaption import SwaptionPricer

if __name__ == "__main__":
    provider = RandomMarketDataProvider(seed=42)
    market = DataLoader(provider).load(currency="USD", asset_class="swaption")

    swap = Swap.from_generator(
        "vanilla",
        notional=1_000_000,
        currency="USD",
        fixed_rate=0.03,
        float_forward_curve=provider.forward_curve_name,
        payment_times_fixed=[1, 2, 3, 4, 5],
        payment_times_float=[1, 2, 3, 4, 5],
        pay_fixed=True,
        spread=0.0,
        day_count=1.0,
    )

    expiry_date = market.as_of + timedelta(days=365)
    swaption = Swaption(swap=swap, expiry=expiry_date, option_type="payer")

    pricer = SwaptionPricer()
    result = pricer.price(swaption, market)

    print("Swaption valuation result using random market data:")
    print(result)