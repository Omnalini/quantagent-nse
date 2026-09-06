"""Shared fixtures. Everything here is offline — no test touches the network."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Tests import the package from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_ohlc(closes, start_ts: int = 1_700_000_000, step: int = 900) -> pd.DataFrame:
    """Build a well-formed OHLC frame from a close series, as nifty_fetcher would."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    ts = start_ts + np.arange(n) * step
    return pd.DataFrame({
        "datetime": pd.to_datetime(ts, unit="s"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(n, 10_000.0),
        "timestamp": ts,
    })


@pytest.fixture
def trending_df():
    """80 bars of a noisy uptrend around ₹1500 — enough for every agent."""
    rng = np.random.default_rng(42)
    closes = 1500 + np.cumsum(rng.normal(0.4, 2.0, 80))
    return make_ohlc(closes)


@pytest.fixture
def falling_df():
    """80 bars of a noisy downtrend around ₹1500."""
    rng = np.random.default_rng(7)
    closes = 1500 + np.cumsum(rng.normal(-0.4, 2.0, 80))
    return make_ohlc(closes)
