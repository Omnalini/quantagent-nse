"""
TrendAgent maths. The report's Section 5.3.1 states the OLS implementation was
checked against scipy.stats.linregress — that check lives here.
"""

import numpy as np
import pytest
from scipy import stats

from quantagent.agents.trend import (
    TrendAgent, find_local_highs, find_local_lows, ols_regression,
)


class TestOLSRegression:
    def test_matches_scipy_linregress_on_random_data(self):
        """Slope, intercept and R² must agree with scipy to ~1e-9."""
        rng = np.random.default_rng(2024)
        for _ in range(30):
            n = rng.integers(10, 200)
            x = np.arange(n, dtype=float)
            y = rng.normal(0, 5, n) + 0.3 * x + 100

            slope, intercept, r2 = ols_regression(x, y)
            ref = stats.linregress(x, y)

            assert slope == pytest.approx(ref.slope, rel=1e-9, abs=1e-9)
            assert intercept == pytest.approx(ref.intercept, rel=1e-9, abs=1e-9)
            assert r2 == pytest.approx(ref.rvalue ** 2, rel=1e-9, abs=1e-9)

    def test_perfect_line_has_r_squared_of_one(self):
        x = np.arange(50, dtype=float)
        slope, intercept, r2 = ols_regression(x, 3.5 * x - 12.0)
        assert slope == pytest.approx(3.5)
        assert intercept == pytest.approx(-12.0)
        assert r2 == pytest.approx(1.0)

    def test_constant_y_hits_the_ss_tot_guard(self):
        """Zero variance in y: must return cleanly, not divide by zero."""
        slope, intercept, r2 = ols_regression(np.arange(20, dtype=float), np.full(20, 7.0))
        assert slope == pytest.approx(0.0)
        assert intercept == pytest.approx(7.0)
        assert r2 == 0.0
        assert not np.isnan(r2)

    def test_constant_x_hits_the_ss_xx_guard(self):
        """Zero variance in x: the ss_xx == 0 guard must return, not raise."""
        slope, intercept, r2 = ols_regression(np.full(20, 4.0), np.arange(20, dtype=float))
        assert not np.isnan(slope)
        assert not np.isnan(intercept)
        assert not np.isnan(r2)

    def test_single_point_returns_safely(self):
        slope, intercept, r2 = ols_regression(np.array([1.0]), np.array([5.0]))
        assert not np.isnan(slope) and not np.isnan(r2)


class TestLocalExtrema:
    """A sine wave has extrema at known indices — the detectors must find them."""

    @staticmethod
    def _sine(cycles=3, n=180, amp=10.0, base=100.0):
        x = np.linspace(0, cycles * 2 * np.pi, n)
        return base + amp * np.sin(x), x

    def test_finds_the_known_peaks(self):
        prices, x = self._sine()
        highs = find_local_highs(prices, window=5)
        assert len(highs) >= 3
        # Every reported high must sit at or beside a true crest of the wave.
        for idx in highs:
            assert prices[idx] > prices.mean()
            local = prices[max(0, idx - 5): idx + 6]
            assert prices[idx] == pytest.approx(local.max(), abs=1e-9)

    def test_finds_the_known_troughs(self):
        prices, _ = self._sine()
        lows = find_local_lows(prices, window=5)
        assert len(lows) >= 3
        for idx in lows:
            assert prices[idx] < prices.mean()
            local = prices[max(0, idx - 5): idx + 6]
            assert prices[idx] == pytest.approx(local.min(), abs=1e-9)

    def test_highs_and_lows_never_coincide(self):
        prices, _ = self._sine()
        assert not (set(find_local_highs(prices, 5).tolist()) &
                    set(find_local_lows(prices, 5).tolist()))

    def test_flat_series_returns_without_raising(self):
        flat = np.full(60, 42.0)
        assert isinstance(find_local_highs(flat, 5), np.ndarray)
        assert isinstance(find_local_lows(flat, 5), np.ndarray)


class TestTrendAgent:
    def test_classifies_a_clear_uptrend(self, trending_df):
        report = TrendAgent().analyze(trending_df)
        assert report.trend in ("Uptrend", "Downtrend", "Sideways")
        assert 0.0 <= report.breakout_probability <= 1.0
        assert 0.0 <= report.resistance_line.r_squared <= 1.0
        assert 0.0 <= report.support_line.r_squared <= 1.0

    def test_kappa_sign_tracks_direction(self, trending_df, falling_df):
        up = TrendAgent().analyze(trending_df)
        down = TrendAgent().analyze(falling_df)
        assert up.kappa > down.kappa
