"""

General Regime Change Model (yfinance edition)
==============================================

Same regime classification as general_regime_model, but uses **yfinance**
as the sole data source - no Bloomberg / BHSysApi dependency.

Outputs:
    - general_regime_output.xlsx  (all core metrics)
    - general_regime_dashboard.html (interactive visualization)

Designed for personal liquid-market investing.
Target: 8-10 distinct regimes, changing roughly monthly/quarterly (>20 changes over 10y).
"""

import os
import sys
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# — 1. DATA FETCHING ————————————————————————————————————————————————————————————————

# Mapping: internal key → (yfinance ticker, scale_factor, description)
# scale_factor is applied after download: value = raw_close * scale_factor
# Yield indices (^IRX, ^FVX, ^TNX, ^TYX) are quoted as yield*10 in yfinance
YF_TICKER_MAP = {
    # Volatility
    'VIX': ('^VIX', 1.0, 'CBOE Volatility Index'),
    'VIX3M': ('^VIX3M', 1.0, 'CBOE 3-Month Volatility Index'),

    # Rates - short, mid, long (yfinance yield indices are *10)
    'US3M': ('^IRX', 0.1, '13-Week Treasury Bill Yield'),
    'US5Y': ('^FVX', 0.1, '5-Year Treasury Yield'),
    'US10Y': ('^TNX', 0.1, '10-Year Treasury Yield'),
    'US30Y': ('^TYX', 0.1, '30-Year Treasury Yield'),

    # Dollar & FX
    'DXY': ('DX-Y.NYB', 1.0, 'US Dollar Index Futures'),

    # Commodities
    'BCOM': ('GSG', 1.0, 'iShares S&P GSCI Commodity ETF (proxy for BCOM)'),
    'OIL': ('CL=F', 1.0, 'WTI Crude Oil Futures'),
    'COPPER': ('HG=F', 1.0, 'Copper Futures'),

    # Credit
    'CDX_HY': ('HYG', 1.0, 'iShares iBoxx High Yield Corporate Bond ETF'),
    'IG_CR': ('LQD', 1.0, 'iShares iBoxx Investment Grade Corporate Bond ETF'),

    # Equity - broad market
    'SPX': ('^GSPC', 1.0, 'S&P 500 Index'),
    'RTY': ('^RUT', 1.0, 'Russell 2000 Index'),
    'MXEF': ('EEM', 1.0, 'iShares MSCI Emerging Markets ETF'),

    # Gold
    'GOLD': ('GC=F', 1.0, 'Gold Futures'),
}

# Portfolio ETFs - used in backtest and recommended allocation
PORTFOLIO_ETF_MAP = {
    'Equities': 'SPY',       # SPDR S&P 500 ETF
    'Long Bonds': 'TLT',     # iShares 20+ Year Treasury Bond ETF
    'Short Dur/Cash': 'BIL', # SPDR 1-3 Month T-Bill ETF
    'Gold': 'GLD',           # SPDR Gold Shares
    'Commodities': 'GSG',    # iShares S&P GSCI Commodity ETF
    'TIPS': 'TIP',           # iShares TIPS Bond ETF
}


def fetch_data(start: str = '2005-01-01') -> pd.DataFrame:
    """Pull all required indicators from yfinance in a single batch."""

    # Collect all tickers (indicator + portfolio ETFs)
    indicator_tickers = [v[0] for v in YF_TICKER_MAP.values()]
    portfolio_tickers = list(PORTFOLIO_ETF_MAP.values())
    all_tickers = list(dict.fromkeys(indicator_tickers + portfolio_tickers))

    print(f"Fetching {len(all_tickers)} tickers from yfinance...")
    print(f"  Indicators: {', '.join(indicator_tickers)}")
    print(f"  Portfolio ETFs: {', '.join(portfolio_tickers)}")

    raw_df = yf.download(all_tickers, start=start, auto_adjust=True, progress=True)

    # yf.download returns MultiIndex columns when >1 ticker: (Price, Ticker)
    # We need the "Close" prices
    if isinstance(raw_df.columns, pd.MultiIndex):
        closes = raw_df['Close']
    else:
        # Single ticker edge case
        closes = raw_df[['Close']].copy()
        closes.columns = all_tickers[:1]

    print(f"\nRaw download: {closes.shape[0]} rows × {closes.shape[1]} tickers")

    # Build the internal series dict with scale factors applied
    series = {}
    for internal_key, (yf_ticker, scale, desc) in YF_TICKER_MAP.items():
        if yf_ticker in closes.columns and closes[yf_ticker].notna().sum() > 100:
            s = closes[yf_ticker].dropna() * scale
            series[internal_key] = s
            print(
                f"  {internal_key:8s} <- {yf_ticker:12s}: {len(s)} rows, "
                f"{s.index.min().date()} to {s.index.max().date()}"
            )
        else:
            print(f"  {internal_key:8s} <- {yf_ticker:12s}: MISSING or insufficient data, skipped")

    # Add portfolio ETF prices under "ETF_<name>" keys
    for ac, ticker in PORTFOLIO_ETF_MAP.items():
        etf_key = f'ETF_{ac}'
        if ticker in closes.columns and closes[ticker].notna().sum() > 100:
            series[etf_key] = closes[ticker].dropna()
            print(f"  {etf_key:20s} <- {ticker:8s}: {len(series[etf_key])} rows")
        else:
            print(f"  {etf_key:20s} <- {ticker:8s}: MISSING")

    if len(series) < 5:
        raise RuntimeError(f"Only fetched {len(series)} series, need at least 5.")

    # Combine into single DataFrame (daily), forward-fill gaps
    combined = pd.DataFrame(series)
    combined = combined.sort_index()
    combined = combined.ffill().dropna()

    print(f"\nCombined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    return combined


def load_cached_or_fetch(cache_path='raw_data.parquet'):
    """Use cached data if fresh (< 1 day old), otherwise re-fetch."""
    full_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_path)
    )
    if os.path.exists(full_path):
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(full_path))
        if (dt.datetime.now() - mtime).days < 1:
            print(f"Using cached data from {full_path}")
            return pd.read_parquet(full_path)

    df = fetch_data()
    df.to_parquet(full_path)
    print(f"Cached data to {full_path}")
    return df


# — 2. FEATURE ENGINEERING ———————————————————————————————————————————————————————————


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    ~40+ macro features covering rates, curves, vol, equity, credit,
    FX, commodities, and cross-asset signals.
    All momentum windows: 1m=21d, 3m=63d, 6m=126d, 12m=252d.

    Note: compared to the Bloomberg version, we lack:
        - BE10Y/BE5Y (breakeven inflation) - no yfinance source
        - FCI (financial conditions) - no yfinance source
        - ISM_PMI (manufacturing PMI) - no yfinance source
        - US2Y (2-year yield) - no reliable yfinance source
    Features guarded by `if "X" in raw.columns` are automatically skipped.
    """
    f = pd.DataFrame(index=raw.index)

    # — RATES & YIELD CURVE —
    if 'US10Y' in raw.columns:
        f['rate_10y_level'] = raw['US10Y']
        f['rate_10y_mom_1m'] = raw['US10Y'] - raw['US10Y'].shift(21)
        f['rate_10y_mom_3m'] = raw['US10Y'] - raw['US10Y'].shift(63)
        f['rate_10y_mom_6m'] = raw['US10Y'] - raw['US10Y'].shift(126)

    if 'US2Y' in raw.columns:
        f['rate_2y_level'] = raw['US2Y']
        f['rate_2y_mom_3m'] = raw['US2Y'] - raw['US2Y'].shift(63)

    if 'US3M' in raw.columns:
        f['rate_3m_level'] = raw['US3M']

    # Yield curve shapes - steepening = growth optimism; inversion = stress
    if 'US10Y' in raw.columns and 'US2Y' in raw.columns:
        f['curve_2s10s'] = raw['US10Y'] - raw['US2Y']
        f['curve_2s10s_chg'] = f['curve_2s10s'] - f['curve_2s10s'].shift(63)

    if 'US10Y' in raw.columns and 'US3M' in raw.columns:
        f['curve_3m10y'] = raw['US10Y'] - raw['US3M']

    if 'US10Y' in raw.columns and 'US5Y' in raw.columns:
        f['curve_5s10s'] = raw['US10Y'] - raw['US5Y']

    if 'US30Y' in raw.columns and 'US10Y' in raw.columns:
        f['curve_10s30s'] = raw['US30Y'] - raw['US10Y']

    # Real yield (nominal - breakeven) - skipped if no breakeven data
    if 'US10Y' in raw.columns and 'BE10Y' in raw.columns:
        f['real_yield_10y'] = raw['US10Y'] - raw['BE10Y']
        f['real_yield_chg_3m'] = f['real_yield_10y'] - f['real_yield_10y'].shift(63)

    # — INFLATION EXPECTATIONS —
    if 'BE10Y' in raw.columns:
        f['infl_exp_10y'] = raw['BE10Y']
        f['infl_exp_chg_1m'] = raw['BE10Y'] - raw['BE10Y'].shift(21)
        f['infl_exp_chg_3m'] = raw['BE10Y'] - raw['BE10Y'].shift(63)
        f['infl_exp_chg_6m'] = raw['BE10Y'] - raw['BE10Y'].shift(126)

    if 'BE5Y' in raw.columns:
        f['infl_exp_5y'] = raw['BE5Y']

    # 5s10s inflation term structure (inflation expectations steepness)
    if 'BE10Y' in raw.columns and 'BE5Y' in raw.columns:
        f['infl_term_5s10s'] = raw['BE10Y'] - raw['BE5Y']

    # — VOLATILITY —
    if 'VIX' in raw.columns:
        f['vix_level'] = raw['VIX']
        f['vix_chg_1m'] = raw['VIX'] - raw['VIX'].shift(21)
        f['vix_chg_3m'] = raw['VIX'] - raw['VIX'].shift(63)

    # VIX term structure: VIX3M/VIX > 1 = contango (calm), < 1 = backwardation (stress)
    if 'VIX3M' in raw.columns and 'VIX' in raw.columns:
        f['vix_term_struct'] = raw['VIX3M'] / raw['VIX'] - 1

    # Realized volatility of equities (21d annualised)
    if 'SPX' in raw.columns:
        spx_ret = raw['SPX'].pct_change()
        f['spx_realvol_1m'] = spx_ret.rolling(21).std() * np.sqrt(252) * 100

    # — EQUITY —
    if 'SPX' in raw.columns:
        f['eq_mom_1m'] = raw['SPX'].pct_change(21) * 100
        f['eq_mom_3m'] = raw['SPX'].pct_change(63) * 100
        f['eq_mom_6m'] = raw['SPX'].pct_change(126) * 100
        f['eq_mom_12m'] = raw['SPX'].pct_change(252) * 100
        f['eq_vs_200d'] = (raw['SPX'] / raw['SPX'].rolling(200).mean() - 1) * 100
        f['eq_vs_50d'] = (raw['SPX'] / raw['SPX'].rolling(50).mean() - 1) * 100

    # Small-cap vs large-cap (risk appetite / cycle phase indicator)
    if 'RTY' in raw.columns and 'SPX' in raw.columns:
        ratio = raw['RTY'] / raw['SPX']
        f['small_vs_large_3m'] = ratio.pct_change(63) * 100

    # EM vs DM equities (global risk appetite)
    if 'MXEF' in raw.columns and 'SPX' in raw.columns:
        em_dm = raw['MXEF'] / raw['SPX']
        f['em_vs_dm_3m'] = em_dm.pct_change(63) * 100

    # — CREDIT —
    if 'CDX_HY' in raw.columns:
        f['hy_mom_1m'] = raw['CDX_HY'].pct_change(21) * 100
        f['hy_mom_3m'] = raw['CDX_HY'].pct_change(63) * 100
        f['hy_vs_200d'] = (raw['CDX_HY'] / raw['CDX_HY'].rolling(200).mean() - 1) * 100

    if 'IG_CR' in raw.columns:
        f['ig_mom_3m'] = raw['IG_CR'].pct_change(63) * 100

    # HY vs IG relative (credit quality rotation)
    if 'CDX_HY' in raw.columns and 'IG_CR' in raw.columns:
        hy_ig = raw['CDX_HY'] / raw['IG_CR']
        f['hy_vs_ig_3m'] = hy_ig.pct_change(63) * 100

    # — DOLLAR & FX —
    if 'DXY' in raw.columns:
        f['dxy_mom_1m'] = raw['DXY'].pct_change(21) * 100
        f['dxy_mom_3m'] = raw['DXY'].pct_change(63) * 100
        f['dxy_mom_6m'] = raw['DXY'].pct_change(126) * 100
        f['dxy_vs_200d'] = (raw['DXY'] / raw['DXY'].rolling(200).mean() - 1) * 100

    # — COMMODITIES —
    if 'BCOM' in raw.columns:
        f['cmdty_mom_1m'] = raw['BCOM'].pct_change(21) * 100
        f['cmdty_mom_3m'] = raw['BCOM'].pct_change(63) * 100
        f['cmdty_mom_6m'] = raw['BCOM'].pct_change(126) * 100

    if 'OIL' in raw.columns:
        f['oil_mom_3m'] = raw['OIL'].pct_change(63) * 100
        f['oil_mom_6m'] = raw['OIL'].pct_change(126) * 100

    if 'COPPER' in raw.columns:
        f['copper_mom_3m'] = raw['COPPER'].pct_change(63) * 100

    if 'GOLD' in raw.columns:
        f['gold_mom_3m'] = raw['GOLD'].pct_change(63) * 100
        f['gold_mom_6m'] = raw['GOLD'].pct_change(126) * 100
        f['gold_vs_200d'] = (raw['GOLD'] / raw['GOLD'].rolling(200).mean() - 1) * 100

    # Gold vs commodities broad (inflation hedge premium)
    if 'GOLD' in raw.columns and 'BCOM' in raw.columns:
        g_b = raw['GOLD'] / raw['BCOM']
        f['gold_vs_cmdty_3m'] = g_b.pct_change(63) * 100

    # — FINANCIAL CONDITIONS & GROWTH —
    if 'FCI' in raw.columns:
        f['fci_level'] = raw['FCI']
        f['fci_chg_3m'] = raw['FCI'] - raw['FCI'].shift(63)

    if 'ISM_PMI' in raw.columns:
        f['ism_pmi_level'] = raw['ISM_PMI']
        f['ism_vs_50'] = raw['ISM_PMI'] - 50
        f['ism_mom_1m'] = raw['ISM_PMI'] - raw['ISM_PMI'].shift(1)
        f['ism_mom_3m'] = raw['ISM_PMI'] - raw['ISM_PMI'].shift(3)

    # — CROSS-ASSET SIGNALS —
    # Equity vs bonds relative momentum (risk-on/off)
    if 'SPX' in raw.columns and 'US10Y' in raw.columns:
        spx_3m = raw['SPX'].pct_change(63) * 100
        bond_3m = -(raw['US10Y'] - raw['US10Y'].shift(63)) * 8  # approx bond return
        f['eq_vs_bonds_3m'] = spx_3m - bond_3m

    # Commodities vs inflation expectations (supply-demand vs expectations gap)
    if 'BCOM' in raw.columns and 'BE10Y' in raw.columns:
        f['cmdty_vs_infl_3m'] = (raw['BCOM'].pct_change(63) * 100)
        f['cmdty_vs_infl_3m'] = f['cmdty_vs_infl_3m'] - ((raw['BE10Y'] - raw['BE10Y'].shift(63)) * 10)

    # Gold vs DXY divergence (stress/safe-haven signal)
    if 'GOLD' in raw.columns and 'DXY' in raw.columns:
        f['gold_dxy_div_3m'] = (raw['GOLD'].pct_change(63) * 100
                                + raw['DXY'].pct_change(63) * 100)  # normally -ve corr

    feat = f.dropna()
    print(f"Features built: {feat.shape[1]} features, {feat.shape[0]} observations")
    return feat


def rolling_zscore(feat: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """
    Compute rolling z-scores (6-month window) for regime sensitivity.
    Shorter window = more regime changes.
    """
    roll_mean = feat.rolling(window, min_periods=window).mean()
    roll_std = feat.rolling(window, min_periods=window).std()
    z = (feat - roll_mean) / roll_std.replace(0, np.nan)
    z = z.dropna()
    print(f"Z-scored features: {z.shape[0]} observations from {z.index.min().date()}")
    return z


# — 3. REGIME CLUSTERING ———————————————————————————————————————————————————————————

N_REGIMES = 9  # Target 9 distinct regimes

# Descriptive labels based on cluster centroid characteristics
REGIME_LABELS = {
    0: 'Risk-On Rally',
    1: 'Goldilocks',
    2: 'Reflation Boom',
    3: 'Tightening Stress',
    4: 'Late-Cycle Grind',
    5: 'Deflationary Shock',
    6: 'Stagflation',
    7: 'Recovery / Early Cycle',
    8: 'Volatility Crisis',
}

# Recommended BASE allocations per regime (for a personal liquid portfolio)
# These are further adjusted by momentum and trend overlays at runtime.
# Categories: Equities, LongDuration Bonds, ShortDuration/Cash, Gold, Commodities, TIPS
REGIME_ALLOCATIONS = {
    'Risk-On Rally': {'Equities': 80, 'Long Bonds': 5, 'Short Dur/Cash': 0, 'Gold': 5, 'Commodities': 5, 'TIPS': 5},
    'Goldilocks': {'Equities': 70, 'Long Bonds': 10, 'Short Dur/Cash': 0, 'Gold': 5, 'Commodities': 10, 'TIPS': 5},
    'Reflation Boom': {'Equities': 55, 'Long Bonds': 0, 'Short Dur/Cash': 5, 'Gold': 10, 'Commodities': 25, 'TIPS': 5},
    'Tightening Stress': {'Equities': 25, 'Long Bonds': 15, 'Short Dur/Cash': 25, 'Gold': 15, 'Commodities': 5, 'TIPS': 15},
    'Late-Cycle Grind': {'Equities': 45, 'Long Bonds': 15, 'Short Dur/Cash': 15, 'Gold': 10, 'Commodities': 10, 'TIPS': 5},
    'Deflationary Shock': {'Equities': 10, 'Long Bonds': 40, 'Short Dur/Cash': 20, 'Gold': 15, 'Commodities': 0, 'TIPS': 15},
    'Stagflation': {'Equities': 15, 'Long Bonds': 5, 'Short Dur/Cash': 20, 'Gold': 25, 'Commodities': 25, 'TIPS': 10},
    'Recovery / Early Cycle': {'Equities': 65, 'Long Bonds': 10, 'Short Dur/Cash': 5, 'Gold': 5, 'Commodities': 10, 'TIPS': 5},
    'Volatility Crisis': {'Equities': 5, 'Long Bonds': 35, 'Short Dur/Cash': 30, 'Gold': 20, 'Commodities': 0, 'TIPS': 10},
}

# — MOMENTUM & TREND OVERLAY PARAMETERS ———————————————————————————————————————————
MOMENTUM_LOOKBACK = 252   # 12-month return
MOMENTUM_SKIP = 21        # skip most recent 1 month (mean-reversion noise)
MOMENTUM_TILT = 0.5       # max 50% relative tilt per asset class
TREND_SMA_WINDOW = 200    # SPX 200-day SMA for trend filter
TREND_BOOST = 0.20        # +20% relative equity boost when above SMA
TREND_CUT = 0.30          # -30% relative equity cut when below SMA

# Exact instruments used for each asset class in the backtest
# (return methodology documented here for full transparency)
PORTFOLIO_INSTRUMENTS = {
    'Equities': {
        'ticker': 'SPY',
        'description': 'SPDR S&P 500 ETF Trust',
        'source': 'yfinance: SPY',
        'return_method': 'ETF adjusted close price return (pct_change).',
    },
    'Long Bonds': {
        'ticker': 'TLT',
        'description': 'iShares 20+ Year Treasury Bond ETF',
        'source': 'yfinance: TLT',
        'return_method': 'ETF adjusted close price return (pct_change).',
    },
    'Short Dur/Cash': {
        'ticker': 'BIL',
        'description': 'SPDR Bloomberg 1-3 Month T-Bill ETF',
        'source': 'yfinance: BIL',
        'return_method': 'ETF adjusted close price return (pct_change).',
    },
    'Gold': {
        'ticker': 'GLD',
        'description': 'SPDR Gold Shares ETF',
        'source': 'yfinance: GLD',
        'return_method': 'ETF adjusted close price return (pct_change).',
    },
    'Commodities': {
        'ticker': 'GSG',
        'description': 'iShares S&P GSCI Commodity-Indexed Trust',
        'source': 'yfinance: GSG',
        'return_method': 'ETF adjusted close price return (pct_change).',
    },
    'TIPS': {
        'ticker': 'TIP',
        'description': 'iShares TIPS Bond ETF',
        'source': 'yfinance: TIP',
        'return_method': 'ETF adjusted close price return (pct_change).',
    },
}


def cluster_regimes(z_features: pd.DataFrame, n_clusters: int = N_REGIMES):
    """Run KMeans clustering on z-scored features to identify regimes."""
    scaler = StandardScaler()
    X = scaler.fit_transform(z_features.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X)

    # Assign descriptive labels based on centroid characteristics
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=z_features.columns,
    )

    # Sort clusters by a composite score to get stable labeling
    # Score = equity momentum - vol level + credit momentum
    score_cols = []
    if 'eq_mom_3m' in centroids.columns:
        score_cols.append('eq_mom_3m')
    if 'vix_level' in centroids.columns:
        centroids['neg_vol'] = -centroids['vix_level']
        score_cols.append('neg_vol')
    elif 'vol_level' in centroids.columns:
        centroids['neg_vol'] = -centroids['vol_level']
        score_cols.append('neg_vol')
    if 'hy_mom_3m' in centroids.columns:
        score_cols.append('hy_mom_3m')
    elif 'credit_mom_3m' in centroids.columns:
        score_cols.append('credit_mom_3m')

    if score_cols:
        centroids['composite'] = centroids[score_cols].mean(axis=1)
    else:
        centroids['composite'] = 0

    rank_order = centroids['composite'].sort_values(ascending=False).index.tolist()

    # Map cluster IDs to regime IDs (0..n-1) by composite rank
    cluster_to_regime = {old: new for new, old in enumerate(rank_order)}
    regime_ids = pd.Series(labels, index=z_features.index).map(cluster_to_regime)

    # Map to label names
    regime_names = regime_ids.map(REGIME_LABELS)

    print(f"\nRegime distribution:")
    for rid in sorted(regime_ids.unique()):
        count = (regime_ids == rid).sum()
        pct = count / len(regime_ids) * 100
        print(f"  {REGIME_LABELS[rid]}: {count:5d} days ({pct:.1f}%)")

    # Count regime changes
    changes = (regime_ids != regime_ids.shift(1)).sum() - 1
    years = (z_features.index[-1] - z_features.index[0]).days / 365.25
    print(f"\nTotal regime changes: {changes} over {years:.1f} years ({changes/years:.1f}/year)")

    return regime_ids, regime_names, centroids, kmeans


def smooth_regimes(regime_ids: pd.Series, min_duration_days: int = 45) -> pd.Series:
    """
    Iteratively remove short-lived regime blips (< min_duration_days).
    Runs multiple passes until stable.
    """
    smoothed = regime_ids.copy()

    for _pass in range(20):  # max 20 passes
        groups = []
        current_val = smoothed.iloc[0]
        start_idx = 0
        for i in range(1, len(smoothed)):
            if smoothed.iloc[i] != current_val:
                groups.append((start_idx, i - 1, current_val))
                current_val = smoothed.iloc[i]
                start_idx = i
        groups.append((start_idx, len(smoothed) - 1, current_val))

        changed = False
        for g_idx in range(len(groups)):
            s, e, val = groups[g_idx]
            duration = (smoothed.index[e] - smoothed.index[s]).days
            if duration < min_duration_days and len(groups) > 1:
                # Replace with previous group's value (or next if first)
                if g_idx > 0:
                    new_val = groups[g_idx - 1][2]
                else:
                    new_val = groups[g_idx + 1][2]
                smoothed.iloc[s:e + 1] = new_val
                changed = True

        if not changed:
            break

    # Recount
    changes = (smoothed != smoothed.shift(1)).sum() - 1
    years = (smoothed.index[-1] - smoothed.index[0]).days / 365.25
    print(f"After smoothing (min {min_duration_days}d): {changes} changes ({changes/years:.1f}/year)")
    return smoothed


# — 4. ASSET RETURNS & BACKTEST —————————————————————————————————————————————————————


def compute_asset_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for each portfolio asset class using ETF prices.
    Much simpler than the Bloomberg version - just pct_change() on ETF prices.
    """
    rets = pd.DataFrame(index=raw.index)

    etf_col_map = {
        'Equities': 'ETF_Equities',
        'Long Bonds': 'ETF_Long Bonds',
        'Short Dur/Cash': 'ETF_Short Dur/Cash',
        'Gold': 'ETF_Gold',
        'Commodities': 'ETF_Commodities',
        'TIPS': 'ETF_TIPS',
    }

    for ac, col in etf_col_map.items():
        if col in raw.columns:
            rets[ac] = raw[col].pct_change()
        else:
            print(f"  WARNING: No {col} data, {ac} return = 0")
            rets[ac] = 0.0

    rets = rets.fillna(0.0)
    # Clip extreme outliers
    rets = rets.clip(-0.10, upper=0.10)
    return rets


def compute_momentum_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-asset 12-1 momentum scores for each portfolio asset class.
    Returns a DataFrame of scores in [-1, +1] range (rank-normalized).
    """
    etf_col_map = {
        'Equities': 'ETF_Equities',
        'Long Bonds': 'ETF_Long Bonds',
        'Short Dur/Cash': 'ETF_Short Dur/Cash',
        'Gold': 'ETF_Gold',
        'Commodities': 'ETF_Commodities',
        'TIPS': 'ETF_TIPS',
    }

    # Compute 12-month return minus 1-month return for each asset
    mom_raw = pd.DataFrame(index=raw.index)
    for ac, col in etf_col_map.items():
        if col in raw.columns:
            ret_12m = raw[col].pct_change(MOMENTUM_LOOKBACK)
            ret_1m = raw[col].pct_change(MOMENTUM_SKIP)
            mom_raw[ac] = ret_12m - ret_1m
        else:
            mom_raw[ac] = 0.0

    # Rank-normalize across assets to [-1, +1] each day
    # rank: 0 (worst) to n-1 (best), then map to [-1, +1]
    ranked = mom_raw.rank(axis=1, pct=True)  # [0, 1] range
    scores = ranked * 2 - 1  # [-1, +1]
    return scores.fillna(0.0)


def compute_trend_signal(raw: pd.DataFrame) -> pd.Series:
    """
    Binary trend signal: +1 if SPX above 200d SMA, -1 if below.
    Returns a Series aligned to raw.index.
    """
    if 'SPX' in raw.columns:
        sma = raw['SPX'].rolling(TREND_SMA_WINDOW, min_periods=TREND_SMA_WINDOW).mean()
        signal = pd.Series(0.0, index=raw.index)
        signal[raw['SPX'] > sma] = 1.0
        signal[raw['SPX'] <= sma] = -1.0
        # NaN for warmup period
        signal[sma.isna()] = 0.0
        return signal
    return pd.Series(0.0, index=raw.index)


def dynamic_allocation(
    regime_names: pd.Series,
    momentum_scores: pd.DataFrame,
    trend_signal: pd.Series,
) -> pd.DataFrame:
    """
    Combine base regime allocation + momentum tilt + trend filter.
    Returns a DataFrame of daily weights (%) for each asset class.
    """
    asset_classes = list(REGIME_ALLOCATIONS[list(REGIME_ALLOCATIONS.keys())[0]].keys())
    idx = regime_names.index
    alloc_daily = pd.DataFrame(0.0, index=idx, columns=asset_classes)

    for date in idx:
        regime = regime_names.loc[date]
        base = REGIME_ALLOCATIONS.get(regime, {})

        # Start with base weights
        weights = {ac: base.get(ac, 0) for ac in asset_classes}

        # — Apply momentum tilt —
        if date in momentum_scores.index:
            for ac in asset_classes:
                if ac in momentum_scores.columns:
                    score = momentum_scores.loc[date, ac]
                    weights[ac] = weights[ac] * (1 + MOMENTUM_TILT * score)

        # — Apply trend filter to equities —
        if date in trend_signal.index:
            ts = trend_signal.loc[date]
            if ts > 0:
                # SPX above SMA: boost equities
                eq_boost = weights['Equities'] * TREND_BOOST
                weights['Equities'] += eq_boost
                # Take from Short Dur/Cash and Long Bonds proportionally
                reduce_from = ['Short Dur/Cash', 'Long Bonds', 'TIPS']
                total_reduce = sum(weights[a] for a in reduce_from)
                if total_reduce > 0:
                    for a in reduce_from:
                        weights[a] -= eq_boost * (weights[a] / total_reduce)
            elif ts < 0:
                # SPX below SMA: cut equities, add to safety
                eq_cut = weights['Equities'] * TREND_CUT
                weights['Equities'] -= eq_cut
                # Add to Short Dur/Cash and Gold
                weights['Short Dur/Cash'] += eq_cut * 0.6
                weights['Gold'] += eq_cut * 0.4

        # — Enforce non-negative and renormalize to 100% —
        for ac in asset_classes:
            weights[ac] = max(weights[ac], 0)
        total = sum(weights.values())
        if total > 0:
            for ac in asset_classes:
                weights[ac] = weights[ac] / total * 100

        for ac in asset_classes:
            alloc_daily.loc[date, ac] = weights[ac]

    return alloc_daily


def backtest_portfolio(
    regime_names: pd.Series,
    asset_returns: pd.DataFrame,
    raw: pd.DataFrame,
):
    """
    Backtest the regime-based allocation (with momentum + trend overlays)
    vs static benchmarks.
    Returns cumulative returns, daily regime return, and daily allocation DataFrame.
    """
    idx = regime_names.index.intersection(asset_returns.index)
    regime_names = regime_names.loc[idx]
    asset_returns = asset_returns.loc[idx]

    asset_classes = list(REGIME_ALLOCATIONS[list(REGIME_ALLOCATIONS.keys())[0]].keys())

    # — Compute overlays —
    print('  Computing momentum scores...')
    momentum_scores = compute_momentum_scores(raw).reindex(idx)
    print('  Computing trend signal...')
    trend_signal = compute_trend_signal(raw).reindex(idx)

    # — Dynamic allocation: regime + momentum + trend —
    print('  Building dynamic allocations...')
    alloc_daily = dynamic_allocation(regime_names, momentum_scores, trend_signal)

    # — Regime-based portfolio with dynamic weights —
    regime_daily_ret = pd.Series(0.0, index=idx)
    for date in idx:
        for ac in asset_classes:
            w = alloc_daily.loc[date, ac] / 100.0
            if ac in asset_returns.columns:
                regime_daily_ret.loc[date] += w * asset_returns.loc[date, ac]

    # — Static 60/40 benchmark (60% equities, 40% bonds)
    static_6040 = pd.Series(0.0, index=idx)
    if 'Equities' in asset_returns.columns and 'Long Bonds' in asset_returns.columns:
        static_6040 = 0.6 * asset_returns['Equities'] + 0.4 * asset_returns['Long Bonds']

    # — Equal-weight benchmark —
    static_equal = asset_returns[asset_classes].mean(axis=1)

    # — 100% Equities benchmark —
    eq_only = asset_returns['Equities'] if 'Equities' in asset_returns.columns else pd.Series(0.0, index=idx)

    # Build cumulative returns
    result = pd.DataFrame(index=idx)
    result['Regime Portfolio'] = (1 + regime_daily_ret).cumprod()
    result['60/40 Benchmark'] = (1 + static_6040).cumprod()
    result['Equal Weight'] = (1 + static_equal).cumprod()
    result['100% Equities'] = (1 + eq_only).cumprod()

    return result, regime_daily_ret, alloc_daily


def compute_backtest_stats(cum_returns: pd.DataFrame, regime_daily_ret: pd.Series) -> pd.DataFrame:
    """Compute annualized return, vol, Sharpe, max drawdown for each strategy."""
    stats = {}
    for col in cum_returns.columns:
        daily_ret = cum_returns[col].pct_change().dropna()
        ann_ret = daily_ret.mean() * 252
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        # Max drawdown
        peak = cum_returns[col].cummax()
        dd = (cum_returns[col] - peak) / peak
        max_dd = dd.min()
        total_ret = cum_returns[col].iloc[-1] / cum_returns[col].iloc[0] - 1
        stats[col] = {
            'Total Return': f'{total_ret:.1%}',
            'Ann. Return': f'{ann_ret:.1%}',
            'Ann. Volatility': f'{ann_vol:.1%}',
            'Sharpe Ratio': f'{sharpe:.2f}',
            'Max Drawdown': f'{max_dd:.1%}',
        }
    return pd.DataFrame(stats)


def build_regime_change_log(regime_names: pd.Series) -> pd.DataFrame:
    """Build a table of all regime transitions with dates and durations."""
    changes = regime_names[regime_names != regime_names.shift(1)]
    rows = []
    for i in range(len(changes)):
        start = changes.index[i]
        end = changes.index[i + 1] if i + 1 < len(changes) else regime_names.index[-1]
        duration = (end - start).days
        rows.append({
            'Start Date': start.strftime('%Y-%m-%d'),
            'End Date': end.strftime('%Y-%m-%d'),
            'Regime': changes.iloc[i],
            'Duration (days)': duration,
        })
    return pd.DataFrame(rows)


def generate_excel(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    regime_ids: pd.Series,
    regime_names: pd.Series,
    centroids: pd.DataFrame,
    cum_returns: pd.DataFrame,
    regime_change_log: pd.DataFrame,
    output_path: str,
):
    """Write all core metrics to Excel."""
    # Daily regime data
    daily = raw.loc[regime_ids.index].copy()
    daily['regime_id'] = regime_ids
    daily['regime_name'] = regime_names

    # Add feature columns
    for col in features.columns:
        if col in features.columns:
            daily[f'z_{col}'] = features[col].reindex(daily.index)

    # Allocations
    alloc_df = pd.DataFrame(REGIME_ALLOCATIONS).T
    alloc_df.index.name = 'regime'

    # Monthly summary
    monthly = daily.resample('ME').last()
    monthly['regime_change'] = (monthly['regime_id'] != monthly['regime_id'].shift(1)).astype(int)

    # Regime transition matrix
    transitions = pd.crosstab(
        regime_names.shift(1).dropna(),
        regime_names.loc[regime_names.shift(1).dropna().index],
        normalize='index',
    ).round(3)

    # Centroids
    centroid_display = centroids.copy()
    centroid_display.index = [REGIME_LABELS.get(i, f'Regime {i}') for i in range(len(centroids))]
    # Drop helper columns
    for c in ['neg_vol', 'composite']:
        if c in centroid_display.columns:
            centroid_display = centroid_display.drop(columns=[c])

    # Instruments reference table
    instruments_df = pd.DataFrame(PORTFOLIO_INSTRUMENTS).T
    instruments_df.index.name = 'Asset Class'

    # Per-regime detailed allocation: each row = regime, columns = instrument + weight
    regime_instrument_rows = []
    for regime, allocs in REGIME_ALLOCATIONS.items():
        for ac, weight in allocs.items():
            info = PORTFOLIO_INSTRUMENTS.get(ac, {})
            regime_instrument_rows.append({
                'Regime': regime,
                'Asset Class': ac,
                'Ticker': info.get('ticker', ''),
                'Description': info.get('description', ''),
                'Weight (%)': weight,
                'Return Methodology': info.get('return_method', ''),
            })
    regime_instrument_df = pd.DataFrame(regime_instrument_rows)

    # yfinance ticker reference
    yf_ref_rows = []
    for key, (ticker, scale, desc) in YF_TICKER_MAP.items():
        yf_ref_rows.append({
            'Internal Key': key,
            'yfinance Ticker': ticker,
            'Scale Factor': scale,
            'Description': desc,
        })
    yf_ref_df = pd.DataFrame(yf_ref_rows)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        daily.to_excel(writer, sheet_name='Daily_Data')
        monthly.to_excel(writer, sheet_name='Monthly_Summary')
        alloc_df.to_excel(writer, sheet_name='Regime_Allocations')
        instruments_df.to_excel(writer, sheet_name='Portfolio_Instruments')
        regime_instrument_df.to_excel(writer, sheet_name='Regime_Instrument_Detail', index=False)
        yf_ref_df.to_excel(writer, sheet_name='yfinance_Ticker_Map', index=False)
        transitions.to_excel(writer, sheet_name='Transition_Matrix')
        centroid_display.to_excel(writer, sheet_name='Regime_Centroids')
        cum_returns.to_excel(writer, sheet_name='Backtest_Returns')
        regime_change_log.to_excel(writer, sheet_name='Regime_Change_Log', index=False)

    print(f"\nExcel output written to {output_path}")


def generate_html(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    regime_ids: pd.Series,
    regime_names: pd.Series,
    cum_returns: pd.DataFrame,
    backtest_stats: pd.DataFrame,
    regime_daily_ret: pd.Series,
    regime_change_log: pd.DataFrame,
    alloc_daily: pd.DataFrame,
    output_path: str,
):
    """Generate interactive HTML dashboard with Plotly."""

    # Color map for regimes
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1
    regime_color_map = {name: colors[i % len(colors)] for i, name in REGIME_LABELS.items()}

    # — Figure 1: Regime timeline with SPX overlay
    fig1 = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.15, 0.35, 0.25, 0.25],
        subplot_titles=['Regime', 'S&P 500 & VIX', '10-Year Yield', 'Equity Allocation %'],
    )

    # Build segments for regime timeline
    segments = []
    current_regime_name = regime_names.iloc[0]
    seg_start = regime_names.index[0]
    for i in range(1, len(regime_names)):
        if regime_names.iloc[i] != current_regime_name:
            segments.append((seg_start, regime_names.index[i], current_regime_name))
            current_regime_name = regime_names.iloc[i]
            seg_start = regime_names.index[i]
    segments.append((seg_start, regime_names.index[-1], current_regime_name))

    # Regime colored background
    for start, end, name in segments:
        color = regime_color_map.get(name, 'gray')
        fig1.add_vrect(
            x0=start, x1=end,
            fillcolor=color, opacity=0.3, line_width=0,
            row=1, col=1,
        )

    # SPX
    if 'SPX' in raw.columns:
        spx = raw['SPX'].loc[regime_ids.index]
        fig1.add_trace(
            go.Scatter(x=spx.index, y=spx.values, name='S&P 500', line=dict(color='blue', width=1.5)),
            row=2, col=1,
        )

    # VIX
    if 'VIX' in raw.columns:
        vix = raw['VIX'].loc[regime_ids.index]
        fig1.add_trace(
            go.Scatter(x=vix.index, y=vix.values, name='VIX', line=dict(color='red', width=1, dash='dot')),
            row=2, col=1,
        )

    # 10Y yield
    if 'US10Y' in raw.columns:
        r10 = raw['US10Y'].loc[regime_ids.index]
        fig1.add_trace(
            go.Scatter(x=r10.index, y=r10.values, name='10Y Yield', line=dict(color='green', width=1, dash='dot')),
            row=3, col=1,
        )

    # Equity allocation over time — dynamic (regime + momentum + trend)
    eq_alloc_dynamic = alloc_daily['Equities'].reindex(regime_ids.index).ffill().fillna(0) if 'Equities' in alloc_daily.columns else pd.Series(0, index=regime_ids.index)
    fig1.add_trace(
        go.Scatter(
            x=eq_alloc_dynamic.index, y=eq_alloc_dynamic.values,
            name='Equity % (dynamic)', fill='tozeroy',
            line=dict(color='darkblue', width=1),
        ),
        row=4, col=1,
    )

    fig1.update_layout(
        height=1100,
        title_text='General Regime Model - Timeline (yfinance)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        template='plotly_white',
    )

    # — Figure 2: Allocation pie for current regime (with instrument labels)
    current_regime = regime_names.iloc[-1]
    current_alloc = REGIME_ALLOCATIONS[current_regime]
    # Label = "Asset Class (Ticker)" for clarity
    pie_labels = [
        f"{ac}<br>({PORTFOLIO_INSTRUMENTS[ac]['ticker']})"
        for ac in current_alloc.keys()
    ]
    pie_hover = [
        f"<b>{ac}</b><br>{PORTFOLIO_INSTRUMENTS[ac]['description']}<br>"
        f"Ticker: {PORTFOLIO_INSTRUMENTS[ac]['ticker']}<br>"
        f"Weight: {w}%<br>"
        f"Return method: {PORTFOLIO_INSTRUMENTS[ac]['return_method']}"
        for ac, w in current_alloc.items()
    ]
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=pie_labels,
                values=list(current_alloc.values()),
                hole=0.4,
                textinfo='label+percent',
                customdata=pie_hover,
                hovertemplate='%{customdata}<extra></extra>',
            )
        ]
    )
    fig_pie.update_layout(
        title_text=f'Current Regime: {current_regime} - Recommended Allocation by Instrument',
        height=500,
        template='plotly_white',
    )

    # — Figure 3: Regime duration histogram
    durations = []
    for start, end, name in segments:
        dur = (end - start).days
        durations.append({'regime': name, 'duration_days': dur})
    dur_df = pd.DataFrame(durations)

    fig_dur = px.histogram(
        dur_df,
        x='duration_days',
        color='regime',
        nbins=40,
        color_discrete_map=regime_color_map,
        title='Regime Duration Distribution (days)',
    )
    fig_dur.update_layout(height=400, template='plotly_white')

    # — Figure 4: Feature heatmap by regime
    feat_by_regime = features.copy()
    feat_by_regime['regime'] = regime_names
    regime_means = feat_by_regime.groupby('regime').mean()

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=regime_means.values,
            x=regime_means.columns.tolist(),
            y=regime_means.index.tolist(),
            colorscale='RdBu_r',
            zmid=0,
            text=regime_means.round(2).values,
            texttemplate='%{text}',
        )
    )
    fig_heat.update_layout(
        title='Average Z-Score by Regime',
        height=500,
        template='plotly_white',
        xaxis_tickangle=-45,
    )

    # — Figure 5: DYNAMIC allocation stacked area over time (regime + momentum + trend)
    asset_classes = ['Equities', 'Long Bonds', 'Short Dur/Cash', 'Gold', 'Commodities', 'TIPS']
    # Use the actual dynamic allocations from the backtest
    alloc_ts = alloc_daily.reindex(regime_ids.index).ffill().fillna(0)
    fig_alloc = go.Figure()
    area_colors = ['#1f77b4', '#2ca02c', '#7f7f7f', '#FFD700', '#d62728', '#9467bd']
    for i, ac in enumerate(asset_classes):
        ticker = PORTFOLIO_INSTRUMENTS.get(ac, {}).get('ticker', '')
        legend_label = f'{ac} ({ticker})' if ticker else ac
        fig_alloc.add_trace(
            go.Scatter(
                x=alloc_ts.index,
                y=alloc_ts[ac] if ac in alloc_ts.columns else [0] * len(alloc_ts),
                name=legend_label,
                stackgroup='one',
                fillcolor=area_colors[i],
                line=dict(width=0.5, color=area_colors[i]),
            )
        )
    fig_alloc.update_layout(
        title='Dynamic Portfolio Allocation Over Time (Regime + Momentum + Trend)',
        yaxis_title='Allocation %',
        height=450,
        template='plotly_white',
        legend=dict(orientation='h'),
    )

    # — Figure 6: Cumulative portfolio returns
    fig_cum = go.Figure()
    line_styles = [
        dict(color='#1a1a2e', width=2.5),
        dict(color='#e63946', width=1.5, dash='dash'),
        dict(color='#457b9d', width=1.5, dash='dot'),
        dict(color='#2a9d8f', width=1.5, dash='dashdot'),
    ]
    for i, col in enumerate(cum_returns.columns):
        fig_cum.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                name=col,
                line=line_styles[i % len(line_styles)],
            )
        )
    fig_cum.update_layout(
        title='Cumulative Portfolio Returns: Regime Model vs Benchmarks',
        yaxis_title='Growth of $1',
        height=500,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        yaxis_type='log',
    )

    # — Figure 7: Rolling 1Y return comparison
    fig_roll = go.Figure()
    for i, col in enumerate(cum_returns.columns):
        daily_ret = cum_returns[col].pct_change()
        roll_1y = (1 + daily_ret).rolling(252).apply(lambda x: x.prod(), raw=True) - 1
        fig_roll.add_trace(
            go.Scatter(
                x=roll_1y.index,
                y=roll_1y.values * 100,
                name=col,
                line=line_styles[i % len(line_styles)],
            )
        )
    fig_roll.update_layout(
        title='Rolling 1-Year Return (%)',
        yaxis_title='Return %',
        height=400,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    # — Figure 8: Drawdown chart for regime portfolio
    peak = cum_returns['Regime Portfolio'].cummax()
    dd = (cum_returns['Regime Portfolio'] - peak) / peak * 100
    peak_eq = cum_returns['100% Equities'].cummax()
    dd_eq = (cum_returns['100% Equities'] - peak_eq) / peak_eq * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name='Regime Portfolio', fill='tozeroy',
                   fillcolor='rgba(26,26,46,0.2)', line=dict(color='#1a1a2e', width=1))
    )
    fig_dd.add_trace(
        go.Scatter(x=dd_eq.index, y=dd_eq.values, name='100% Equities', fill='tozeroy',
                   fillcolor='rgba(230,57,70,0.15)', line=dict(color='#e63946', width=1, dash='dash'))
    )
    fig_dd.update_layout(
        title='Drawdown: Regime Portfolio vs 100% Equities',
        yaxis_title='Drawdown %',
        height=350,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    # — Figure 9: Single-row regime timeline strip
    fig_gantt = go.Figure()

    for idx_seg, (start, end, name) in enumerate(segments):
        mid = start + (end - start) / 2
        dur = (end - start).days
        color = regime_color_map.get(name, 'gray')
        fig_gantt.add_trace(
            go.Scatter(
                x=[mid],
                y=[0.5],
                mode='markers',
                marker=dict(size=0.1, color='rgba(0,0,0,0)'),
                name=name,
                showlegend=(name not in [s[2] for s in segments[:idx_seg]]),
                legendgroup=name,
                hovertemplate=(
                    f'<b>{name}</b><br>'
                    f'{start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}<br>'
                    f'Duration: {dur} days<extra></extra>'
                ),
            )
        )

    shapes = []
    for start, end, name in segments:
        color = regime_color_map.get(name, 'gray')
        shapes.append(dict(
            type='rect',
            x0=start, x1=end,
            y0=0, y1=1,
            fillcolor=color,
            opacity=0.85,
            line=dict(width=0),
            layer='below',
        ))

    fig_gantt.update_layout(
        title='Regime Timeline',
        shapes=shapes,
        height=150,
        template='plotly_white',
        xaxis=dict(type='date', title=''),
        yaxis=dict(visible=False, range=[0, 1]),
        legend=dict(orientation='h', yanchor='bottom', y=1.08, font=dict(size=11)),
        margin=dict(l=20, r=20, t=80, b=30),
    )

    # — Build regime legend HTML
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:16px 0;">'
    for rid, name in REGIME_LABELS.items():
        c = regime_color_map[name]
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;gap:4px;">'
            f'<span style="display:inline-block;width:14px;height:14px;background:{c};'
            f'border-radius:3px;"></span>{name}</span>'
        )
    legend_html += '</div>'

    # — Assemble HTML
    html_parts = [
        '<!DOCTYPE html><html><head>',
        '<meta charset="utf-8">',
        '<title>General Regime Model Dashboard (yfinance)</title>',
        '<style>',
        'body{font-family:system-ui,-apple-system,sans-serif;margin:20px;background:#fafafa;color:#333;}',
        'h1{color:#1a1a2e;} h2{color:#16213e;margin-top:40px;}',
        '.summary{background:#fff;border-radius:8px;padding:20px;margin:16px 0;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.08);display:flex;flex-wrap:wrap;gap:24px;overflow-x:auto;}',
        '.metric{text-align:center;min-width:140px;}',
        '.metric .val{font-size:28px;font-weight:bold;color:#1a1a2e;}',
        '.metric .lbl{font-size:12px;color:#888;margin-top:4px;}',
        'table{border-collapse:collapse;} th,td{padding:6px 12px;text-align:left;}',
        '</style>',
        '</head><body>',
        '<h1>General Regime Model Dashboard (yfinance)</h1>',
        f'<p>Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M")} &mdash; '
        f'Data source: <b>yfinance</b> (no Bloomberg dependency)</p>',
        legend_html,
        '<div class="summary">',
        f"<div class='metric'><div class='val'>{current_regime}</div><div class='lbl'>Current Regime</div></div>",
        f"<div class='metric'><div class='val'>{len(segments)}</div><div class='lbl'>Total Regime Changes</div></div>",
        f"<div class='metric'><div class='val'>{len(segments)/((regime_ids.index[-1]-regime_ids.index[0]).days/365.25):.1f}/yr</div><div class='lbl'>Avg Changes/Year</div></div>",
        f"<div class='metric'><div class='val'>{regime_ids.nunique()}</div><div class='lbl'>Distinct Regimes</div></div>",
        '</div>',
    ]

    # Current allocation table — merged with instrument detail for full transparency
    html_parts.append(f'<h2>Current Recommended Allocation — {current_regime}</h2>')
    html_parts.append(
        '<p>Each row shows the exact ETF used, its allocation weight, and how returns are computed in the backtest.</p>'
    )

    html_parts.append(
        '<div class="summary"><table style="border-collapse:collapse;width:100%;font-size:13px;">'
        '<thead><tr>'
        '<th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">Asset Class</th>'
        '<th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">ETF (Ticker)</th>'
        '<th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">Description</th>'
        '<th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:center;background:#f5f5f5;">Weight/t</th>'
        '<th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">Return Methodology</th>'
        '</tr></thead><tbody>'
    )
    for ac, info in PORTFOLIO_INSTRUMENTS.items():
        weight = current_alloc.get(ac, 0)
        bar_width = max(weight, 0)
        html_parts.append(
            f'<tr>'
            f'<td style="padding:7px 12px;font-weight:bold;border-bottom:1px solid #eee;">{ac}</td>'
            f'<td style="padding:7px 12px;font-family:monospace;border-bottom:1px solid #eee;">{info["ticker"]}</td>'
            f'<td style="padding:7px 12px;border-bottom:1px solid #eee;">{info["description"]}</td>'
            f'<td style="padding:7px 12px;text-align:center;font-weight:bold;border-bottom:1px solid #eee;">'
            f'<div style="display:flex;align-items:center;justify-content:center;gap:6px;">'
            f'<div style="background:#1a1a2e;height:14px;width:{bar_width * 2}px;border-radius:2px;"></div>'
            f'{weight}%</div></td>'
            f'<td style="padding:7px 12px;color:#555;border-bottom:1px solid #eee;font-size:12px;">{info["return_method"]}</td>'
            f'</tr>'
        )
    html_parts.append('</tbody></table></div>')

    # — All regime allocations table (collapsed by default)
    html_parts.append('<h2>All Regime Allocations by Instrument</h2>')
    html_parts.append(
        '<details><summary style="cursor:pointer;font-weight:bold;padding:8px 0;">Click to expand full allocation table for all regimes</summary>'
    )
    html_parts.append(
        '<div class="summary"><table style="border-collapse:collapse;width:100%;font-size:12px;">'
        '<thead><tr>'
        '<th style="padding:6px 10px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">Regime</th>'
    )
    for ac in PORTFOLIO_INSTRUMENTS:
        ticker = PORTFOLIO_INSTRUMENTS[ac]['ticker']
        html_parts.append(
            f'<th style="padding:6px 10px;border-bottom:2px solid #ddd;text-align:center;background:#f5f5f5;">'
            f'{ac}<br><span style="font-family:monospace;font-size:10px;color:#666;">{ticker}</span></th>'
        )
    html_parts.append('</tr></thead><tbody>')
    for regime, allocs in REGIME_ALLOCATIONS.items():
        is_current = regime == current_regime
        row_style = 'background:#e8f4fd;font-weight:bold;' if is_current else ''
        html_parts.append(f'<tr style="{row_style}">')
        label = f'{regime} (CURRENT)' if is_current else regime
        html_parts.append(f'<td style="padding:5px 10px;border-bottom:1px solid #eee;white-space:nowrap;">{label}</td>')
        for ac in PORTFOLIO_INSTRUMENTS:
            w = allocs.get(ac, 0)
            html_parts.append(
                f'<td style="padding:5px 10px;text-align:center;border-bottom:1px solid #eee;">{w}%</td>'
            )
        html_parts.append('</tr>')
    html_parts.append('</tbody></table></div></details>')

    # — yfinance data source reference
    html_parts.append('<h2>Data Sources (yfinance Tickers)</h2>')
    html_parts.append(
        '<details><summary style="cursor:pointer;font-weight:bold;padding:8px 0;">Click to expand yfinance ticker mapping</summary>'
    )
    html_parts.append(
        '<div class="summary"><table style="border-collapse:collapse;width:100%;font-size:12px;">'
        '<thead><tr>'
        '<th style="padding:6px 10px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">Internal Key</th>'
        '<th style="padding:6px 10px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">yfinance Ticker</th>'
        '<th style="padding:6px 10px;border-bottom:2px solid #ddd;text-align:center;background:#f5f5f5;">Scale</th>'
        '<th style="padding:6px 10px;border-bottom:2px solid #ddd;text-align:left;background:#f5f5f5;">Description</th>'
        '</tr></thead><tbody>'
    )
    for key, (ticker, scale, desc) in YF_TICKER_MAP.items():
        available = key in raw.columns
        status = '&#9989;' if available else '&#10060;'
        html_parts.append(
            f'<tr>'
            f'<td style="padding:5px 10px;font-family:monospace;border-bottom:1px solid #eee;">{key}</td>'
            f'<td style="padding:5px 10px;font-family:monospace;border-bottom:1px solid #eee;">{ticker}</td>'
            f'<td style="padding:5px 10px;text-align:center;border-bottom:1px solid #eee;">{scale}</td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #eee;">{desc} {status}</td>'
            f'</tr>'
        )
    html_parts.append('</tbody></table></div></details>')

    # — Section: Regime timeline + market charts
    html_parts.append('<h2>Regime Timeline & Market Context</h2>')
    html_parts.append(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    html_parts.append(fig_pie.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append(fig_alloc.to_html(full_html=False, include_plotlyjs=False))

    # — Section: Regime Change Log (visual + table)
    html_parts.append('<h2>Regime Change History</h2>')
    html_parts.append(fig_gantt.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('<p>Complete log of all regime transitions. Each row shows when the regime changed and how long it lasted.</p>')
    html_parts.append('<div class="summary" style="max-height:600px;overflow-y:auto;">')
    html_parts.append('<table style="border-collapse:collapse;width:100%;font-size:13px;">')
    html_parts.append(
        '<thead style="position:sticky;top:0;background:#fff;"><tr>'
        '<th style="padding:6px 12px;border-bottom:2px solid #ddd;text-align:left;">#</th>'
        '<th style="padding:6px 12px;border-bottom:2px solid #ddd;text-align:left;">Start Date</th>'
        '<th style="padding:6px 12px;border-bottom:2px solid #ddd;text-align:left;">End Date</th>'
        '<th style="padding:6px 12px;border-bottom:2px solid #ddd;text-align:left;">Regime</th>'
        '<th style="padding:6px 12px;border-bottom:2px solid #ddd;text-align:right;">Duration (days)</th>'
        '</tr></thead><tbody>'
    )
    for i, row in regime_change_log.iterrows():
        color = regime_color_map.get(row['Regime'], '#fff')
        html_parts.append(
            f'<tr style="background:linear-gradient(90deg,{color}22 0%,transparent 40%);">'
            f'<td style="padding:4px 12px;">{i+1}</td>'
            f'<td style="padding:4px 12px;">{row["Start Date"]}</td>'
            f'<td style="padding:4px 12px;">{row["End Date"]}</td>'
            f'<td style="padding:4px 12px;font-weight:bold;">'
            f'<span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px;"></span>'
            f'{row["Regime"]}</td>'
            f'<td style="padding:4px 12px;text-align:right;">{row["Duration (days)"]}</td>'
            f'</tr>'
        )
    html_parts.append('</tbody></table></div>')

    # — Section: Portfolio Backtest
    html_parts.append('<h2>Portfolio Backtest: Cumulative Returns</h2>')
    html_parts.append(
        '<p>Simulated performance of following the regime-based allocation vs static benchmarks. '
        'Rebalanced at each regime change. ETFs used: '
        + ', '.join(
            f'<b>{ac}</b> = {info["ticker"]} ({info["description"]})'
            for ac, info in PORTFOLIO_INSTRUMENTS.items()
        )
        + '.</p>'
    )

    # Backtest stats table
    html_parts.append('<div class="summary">')
    html_parts.append('<table style="border-collapse:collapse;width:100%;">')
    html_parts.append('<tr><th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:left;">Metric</th>')
    for col in backtest_stats.columns:
        html_parts.append(f'<th style="padding:8px 12px;border-bottom:2px solid #ddd;text-align:right;">{col}</th>')
    html_parts.append('</tr>')
    for metric in backtest_stats.index:
        html_parts.append(f'<tr><td style="padding:6px 12px;font-weight:bold;">{metric}</td>')
        for col in backtest_stats.columns:
            val = backtest_stats.loc[metric, col]
            html_parts.append(f'<td style="padding:6px 12px;text-align:right;">{val}</td>')
        html_parts.append('</tr>')
    html_parts.append('</table></div>')

    html_parts.append(fig_cum.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append(fig_roll.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append(fig_dd.to_html(full_html=False, include_plotlyjs=False))

    # — Section: Regime analytics
    html_parts.append('<h2>Regime Analytics</h2>')
    html_parts.append(fig_heat.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append(fig_dur.to_html(full_html=False, include_plotlyjs=False))

    html_parts.append('</body></html>')

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(''.join(html_parts))

    print(f"HTML dashboard written to {output_path}")


# — 5. MAIN ——————————————————————————————————————————————————————————————————————————


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Get data
    raw = load_cached_or_fetch()

    # Step 2: Build features
    features = build_features(raw)

    # Step 3: Rolling z-scores on daily data
    z_features_daily = rolling_zscore(features, window=126)

    # Step 4: Resample to MONTHLY for clustering (avoids noisy daily whipsaw)
    z_monthly = z_features_daily.resample('ME').mean().dropna()
    print(f"Monthly z-scored features: {z_monthly.shape[0]} months")

    # Step 5: Cluster on monthly data
    regime_ids_monthly, regime_names_monthly, centroids, kmeans = cluster_regimes(
        z_monthly, n_clusters=N_REGIMES
    )

    # Step 6: Smooth — remove regimes lasting < 1 month
    regime_ids_monthly_smooth = smooth_regimes(regime_ids_monthly, min_duration_days=25)
    regime_names_monthly_smooth = regime_ids_monthly_smooth.map(REGIME_LABELS)

    # Step 7: Expand monthly regimes back to daily (forward-fill)
    regime_ids_daily = regime_ids_monthly_smooth.reindex(z_features_daily.index, method='ffill')
    regime_names_daily = regime_ids_daily.map(REGIME_LABELS)
    # Fill any leading NaNs
    regime_ids_daily = regime_ids_daily.bfill()
    regime_names_daily = regime_ids_daily.map(REGIME_LABELS)

    # Step 8: Compute asset returns and backtest (with momentum + trend overlays)
    print("\nRunning portfolio backtest...")
    asset_returns = compute_asset_returns(raw)
    cum_returns, regime_daily_ret, alloc_daily = backtest_portfolio(
        regime_names_daily, asset_returns, raw
    )
    backtest_stats = compute_backtest_stats(cum_returns, regime_daily_ret)
    regime_change_log = build_regime_change_log(regime_names_daily)

    print(f"\nBacktest period: {cum_returns.index[0].date()} to {cum_returns.index[-1].date()}")
    print(backtest_stats.to_string())

    # Step 9: Generate outputs
    excel_path = os.path.join(out_dir, 'general_regime_output.xlsx')
    generate_excel(raw, z_features_daily, regime_ids_daily, regime_names_daily, centroids,
                   cum_returns, regime_change_log, excel_path)

    html_path = os.path.join(out_dir, 'general_regime_dashboard.html')
    generate_html(
        raw, z_features_daily, regime_ids_daily, regime_names_daily,
        cum_returns, backtest_stats, regime_daily_ret, regime_change_log,
        alloc_daily, html_path,
    )

    current_regime = regime_names_daily.iloc[-1]
    base_alloc = REGIME_ALLOCATIONS[current_regime]
    # Get the actual dynamic allocation for the latest day
    latest_dynamic = alloc_daily.iloc[-1].to_dict() if len(alloc_daily) > 0 else base_alloc
    print(f"\nCurrent regime: {current_regime}")
    print(f"\nRecommended portfolio allocation (dynamic: regime + momentum + trend):")
    print(f"  {'Asset Class':<18} {'ETF Ticker':<12} {'Description':<40} {'Base':>5} {'Dynamic':>8}")
    print(f"  {'-'*18} {'-'*12} {'-'*40} {'-'*5} {'-'*8}")
    for ac in base_alloc:
        info = PORTFOLIO_INSTRUMENTS.get(ac, {})
        ticker = info.get('ticker', 'N/A')
        desc = info.get('description', '')
        base_w = base_alloc[ac]
        dyn_w = latest_dynamic.get(ac, base_w)
        print(f"  {ac:<18} {ticker:<12} {desc:<40} {base_w:>4}% {dyn_w:>7.1f}%")
    print(f"  {'-'*18} {'-'*12} {'-'*40} {'-'*5} {'-'*8}")
    print(f"  {'TOTAL':<18} {'':<12} {'':<40} {sum(base_alloc.values()):>4}% {sum(latest_dynamic.get(ac, 0) for ac in base_alloc):>7.1f}%")
    print("\nDone!")


if __name__ == '__main__':
    main()
