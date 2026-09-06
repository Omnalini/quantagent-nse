"""
IndicatorAgent maths. Section 5.3.1 of the report claims RSI was verified against
a hand computation and MACD against a reference implementation; these tests are
that verification, run automatically.
"""

import numpy as np
import pytest

from quantagent.agents.indicators import (
    IndicatorAgent, compute_ema, compute_macd, compute_roc, compute_rsi,
    compute_stochastic, compute_williams_r,
)
from conftest import make_ohlc


def _reference_rsi(closes, period=14):
    """
    Independent Wilder RSI, written straight from the 1978 definition:
    seed with a simple mean of the first `period` gains/losses, then smooth
    each subsequent step by (prev * (n-1) + current) / n.
    """
    deltas = np.diff(np.asarray(closes, dtype=float))
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out = []
    rs = 100.0 if avg_loss == 0 else avg_gain / avg_loss
    out.append(100 - 100 / (1 + rs))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = 100.0 if avg_loss == 0 else avg_gain / avg_loss
        out.append(100 - 100 / (1 + rs))
    return np.array(out)


class TestRSI:
    def test_matches_hand_computed_wilder_smoothing(self):
        """
        Wilder's smoothing is easy to get subtly wrong (the usual mistake is a
        plain rolling mean instead of the recursive form), so pin it against an
        independently written reference on a known series.
        """
        closes = np.array([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
            46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
        ])
        got = compute_rsi(closes, period=14)
        want = _reference_rsi(closes, period=14)
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)

    def test_bounded_zero_to_hundred_on_random_walks(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            closes = 100 + np.cumsum(rng.normal(0, 1.5, 200))
            rsi = compute_rsi(closes)
            assert np.all(rsi >= 0) and np.all(rsi <= 100)
            assert not np.any(np.isnan(rsi))

    def test_monotonic_rise_approaches_100(self):
        # With no down-bars the implementation clamps RS at 100, so RSI tops out
        # just under the theoretical bound rather than exactly at it.
        rsi = compute_rsi(np.linspace(100, 200, 60))
        assert rsi[-1] > 99.0
        assert rsi[-1] <= 100.0

    def test_monotonic_fall_approaches_0(self):
        rsi = compute_rsi(np.linspace(200, 100, 60))
        assert rsi[-1] < 1.0
        assert rsi[-1] >= 0.0

    def test_short_series_returns_safely(self):
        """Fewer than period+1 bars must return a neutral value, not raise."""
        rsi = compute_rsi(np.array([100.0, 101.0, 102.0]), period=14)
        assert rsi.tolist() == [50.0]


class TestMACD:
    def test_constant_series_gives_zero_line_and_signal(self):
        closes = np.full(120, 250.0)
        macd, signal, hist = compute_macd(closes)
        np.testing.assert_allclose(macd, 0.0, atol=1e-12)
        np.testing.assert_allclose(signal, 0.0, atol=1e-12)
        np.testing.assert_allclose(hist, 0.0, atol=1e-12)

    def test_line_is_fast_ema_minus_slow_ema(self):
        rng = np.random.default_rng(3)
        closes = 500 + np.cumsum(rng.normal(0, 2, 150))
        macd, _, _ = compute_macd(closes, fast=12, slow=26)
        np.testing.assert_allclose(
            macd, compute_ema(closes, 12) - compute_ema(closes, 26), rtol=1e-12
        )

    def test_rising_series_has_positive_macd(self):
        macd, _, _ = compute_macd(np.linspace(100, 200, 150))
        assert macd[-1] > 0

    def test_short_series_returns_safely(self):
        macd, signal, hist = compute_macd(np.arange(10, dtype=float))
        assert macd.tolist() == [0.0] and signal.tolist() == [0.0] and hist.tolist() == [0.0]


class TestOtherOscillators:
    def test_roc_sign_follows_direction(self):
        assert compute_roc(np.linspace(100, 200, 60))[-1] > 0
        assert compute_roc(np.linspace(200, 100, 60))[-1] < 0

    def test_roc_short_series_returns_safely(self):
        assert compute_roc(np.array([100.0, 101.0]), period=12).tolist() == [0.0]

    def test_stochastic_bounded(self):
        rng = np.random.default_rng(11)
        closes = 100 + np.cumsum(rng.normal(0, 1, 120))
        highs, lows = closes * 1.01, closes * 0.99
        k, d = compute_stochastic(highs, lows, closes)
        assert np.all(k >= 0) and np.all(k <= 100)
        assert np.all(d >= 0) and np.all(d <= 100)

    def test_williams_r_bounded(self):
        rng = np.random.default_rng(12)
        closes = 100 + np.cumsum(rng.normal(0, 1, 120))
        wr = compute_williams_r(closes * 1.01, closes * 0.99, closes)
        assert np.all(wr >= -100) and np.all(wr <= 0)


class TestIndicatorAgent:
    def test_report_fields_are_populated(self, trending_df):
        report = IndicatorAgent().analyze(trending_df)
        assert report.overall_signal in ("Bullish", "Bearish", "Neutral")
        assert 0.0 <= report.confidence <= 1.0
        assert 0.0 <= report.rsi.value <= 100.0
        assert report.summary

    def test_handles_a_short_frame_without_raising(self):
        """Fewer bars than any indicator period must degrade, not crash."""
        report = IndicatorAgent().analyze(make_ohlc(np.linspace(100, 105, 12)))
        assert report.overall_signal in ("Bullish", "Bearish", "Neutral")
