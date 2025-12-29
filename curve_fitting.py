"""Example: customized US swap curve fitting from tenors and rate type."""
from __future__ import annotations

from datetime import date
from typing import List

import matplotlib.pyplot as plt

from qfinlib.calibration.curve.builder import CurveBuilder
from qfinlib.market.data.random_provider import RandomMarketDataProvider


def build_us_swap_curve(tenors: List[str], rate_type: str = "listed"):
    """Construct a synthetic US swap curve and return curve plus input data."""

    provider = RandomMarketDataProvider(seed=123, currency="USD")
    market = provider.build_market(currency="USD", as_of=date(2024, 1, 2))

    swap_quotes = provider.generate_swap_curve(tenors, rate_type=rate_type)
    builder = CurveBuilder(market)
    swap_curve = builder.build_discount_curve(
        currency="USD",
        instruments=swap_quotes,
        interpolation="monotone",
        extrapolation="flat",
    )
    return swap_curve, swap_quotes


def plot_swap_curve(curve, swap_quotes, rate_type: str):
    grid = [i / 12 for i in range(1, int(max(q["pillar"] for q in swap_quotes) * 12) + 1)]
    zero_rates = [curve.zero_rate(t) for t in grid]

    plt.figure(figsize=(8, 4))
    plt.plot(grid, zero_rates, label="Interpolated zero curve")
    plt.scatter(
        [q["pillar"] for q in swap_quotes],
        [q["quote"] for q in swap_quotes],
        color="red",
        label="Input swap quotes",
    )
    plt.title(f"US Swap Curve ({rate_type.upper()})")
    plt.xlabel("Tenor (years)")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    us_swap_tenors = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    rate_type = "listed"  # or "otc"

    curve, quotes = build_us_swap_curve(us_swap_tenors, rate_type=rate_type)
    print("Input quotes:")
    for quote in quotes:
        print(quote)
    plot_swap_curve(curve, quotes, rate_type)