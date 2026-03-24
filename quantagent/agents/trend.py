"""
TrendAgent: Slope-Aware Trend Detection Module
Replicates QuantAgent TrendAgent from the paper (Section 3.1.2).
Implements Algorithm 1: OLS regression on highs/lows for support/resistance.
Classifies: Uptrend, Downtrend, Sideways
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class TrendLine:
    slope: float
    intercept: float
    r_squared: float
    direction: str      # "Upward", "Downward", "Flat"
    strength: str       # "Strong", "Moderate", "Weak"
    label: str          # "Resistance" or "Support"


@dataclass
class TrendReport:
    trend: str                  # "Uptrend", "Downtrend", "Sideways"
    kappa: float                # Average slope (mr + ms) / 2 from paper Algorithm 1
    resistance_line: TrendLine
    support_line: TrendLine
    momentum_strength: str      # "Strong", "Moderate", "Weak"
    prediction: str
    reasoning: str
    signals: str
    channel_width: float
    breakout_probability: float
    pivot_highs: List[Tuple[int, float]] = field(default_factory=list)
    pivot_lows: List[Tuple[int, float]] = field(default_factory=list)


# Threshold τ for trend classification (from paper Algorithm 1)
DEFAULT_TAU = 0.0001


def ols_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Ordinary Least Squares regression.
    Returns: slope (m), intercept (b), R-squared
    From paper Algorithm 1: fit OLS on highs/lows to get mr, ms
    """
    if len(x) < 2:
        return 0.0, float(np.mean(y)) if len(y) > 0 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return 0.0, float(y_mean), 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return float(slope), float(intercept), float(max(0, r2))


def find_local_highs(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """Find indices of local highs"""
    highs = []
    for i in range(window, len(prices) - window):
        if prices[i] == np.max(prices[i - window:i + window + 1]):
            highs.append(i)
    return np.array(highs) if highs else np.array([0, len(prices) - 1])


def find_local_lows(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """Find indices of local lows"""
    lows = []
    for i in range(window, len(prices) - window):
        if prices[i] == np.min(prices[i - window:i + window + 1]):
            lows.append(i)
    return np.array(lows) if lows else np.array([0, len(prices) - 1])


def classify_slope_direction(slope: float, price_level: float) -> Tuple[str, str]:
    """Classify slope as upward/downward/flat and assess strength"""
    if price_level == 0:
        return "Flat", "Weak"
    # Normalize slope relative to price
    norm = abs(slope) / price_level

    if slope > 0:
        direction = "Upward"
    elif slope < 0:
        direction = "Downward"
    else:
        direction = "Flat"

    if norm > 0.001:
        strength = "Strong"
    elif norm > 0.0003:
        strength = "Moderate"
    else:
        strength = "Weak"

    return direction, strength


class TrendAgent:
    """
    TrendAgent: tracks direction and steepness of price movements.
    Implements paper Algorithm 1:
    - Fits OLS on recent highs → resistance line R_t(x) = mr*x + br
    - Fits OLS on recent lows  → support line S_t(x) = ms*x + bs
    - κ_t = (mr + ms) / 2 (average slope)
    - Classifies: Uptrend if κ > τ, Downtrend if κ < -τ, else Sideways
    """

    def __init__(self, window: int = 20, tau: float = DEFAULT_TAU):
        self.name = "TrendAgent"
        self.window = window
        self.tau = tau

    def analyze(self, ohlc_df: pd.DataFrame) -> TrendReport:
        """
        Main trend analysis. Takes OHLC DataFrame.
        Returns TrendReport with trendlines, classification, and summaries.
        """
        closes = ohlc_df['close'].values.astype(float)
        highs = ohlc_df['high'].values.astype(float)
        lows = ohlc_df['low'].values.astype(float)

        N = min(self.window, len(closes))
        recent_closes = closes[-N:]
        recent_highs = highs[-N:]
        recent_lows = lows[-N:]
        x_idx = np.arange(N, dtype=float)

        # Algorithm 1: fit OLS on highs (resistance) and lows (support)
        mr, br, r2_r = ols_regression(x_idx, recent_highs)
        ms, bs, r2_s = ols_regression(x_idx, recent_lows)

        # κ_t = (mr + ms) / 2
        kappa = (mr + ms) / 2.0

        # Trend classification per Algorithm 1
        # Normalize tau by price level
        price_level = np.mean(recent_closes)
        norm_kappa = kappa / price_level if price_level != 0 else kappa

        if norm_kappa > self.tau:
            trend = "Uptrend"
        elif norm_kappa < -self.tau:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        # Build trendline objects
        r_dir, r_str = classify_slope_direction(mr, price_level)
        s_dir, s_str = classify_slope_direction(ms, price_level)

        resistance = TrendLine(
            slope=mr, intercept=br, r_squared=r2_r,
            direction=r_dir, strength=r_str, label="Resistance"
        )
        support = TrendLine(
            slope=ms, intercept=bs, r_squared=r2_s,
            direction=s_dir, strength=s_str, label="Support"
        )

        # Channel width (distance between lines at last point)
        r_end = mr * (N - 1) + br
        s_end = ms * (N - 1) + bs
        channel_width = abs(r_end - s_end)

        # Breakout probability: tighter channel = higher breakout probability
        max_range = np.max(recent_highs) - np.min(recent_lows)
        breakout_prob = min(0.95, 1.0 - (channel_width / max_range)) if max_range > 0 else 0.5

        # Momentum strength
        momentum = self._assess_momentum(trend, kappa, price_level, r2_r, r2_s)

        # Generate structured text summaries
        prediction, reasoning, signals = self._generate_summaries(
            trend, resistance, support, recent_closes, channel_width, breakout_prob
        )

        # Pivot points for visualization
        h_win = max(3, N // 10)
        pivot_high_idx = find_local_highs(recent_highs, window=h_win)
        pivot_low_idx = find_local_lows(recent_lows, window=h_win)
        offset = len(closes) - N
        pivot_highs = [(int(i + offset), float(recent_highs[i])) for i in pivot_high_idx]
        pivot_lows = [(int(i + offset), float(recent_lows[i])) for i in pivot_low_idx]

        return TrendReport(
            trend=trend,
            kappa=kappa,
            resistance_line=resistance,
            support_line=support,
            momentum_strength=momentum,
            prediction=prediction,
            reasoning=reasoning,
            signals=signals,
            channel_width=channel_width,
            breakout_probability=breakout_prob,
            pivot_highs=pivot_highs,
            pivot_lows=pivot_lows
        )

    def _assess_momentum(self, trend: str, kappa: float, price_level: float,
                          r2_r: float, r2_s: float) -> str:
        """Assess overall momentum strength"""
        avg_r2 = (r2_r + r2_s) / 2
        norm_k = abs(kappa) / price_level if price_level != 0 else 0

        if avg_r2 > 0.7 and norm_k > 0.001:
            return "Strong"
        elif avg_r2 > 0.4 or norm_k > 0.0003:
            return "Moderate"
        else:
            return "Weak"

    def _generate_summaries(self, trend: str, resistance: TrendLine,
                             support: TrendLine, closes: np.ndarray,
                             channel_width: float,
                             breakout_prob: float) -> Tuple[str, str, str]:
        """
        Generate the three structured summaries shown in paper Figure 1 (TrendAgent):
        Prediction, Reasoning, Signals
        """
        last_close = float(closes[-1])
        r_val = resistance.slope * (len(closes) - 1) + resistance.intercept
        s_val = support.slope * (len(closes) - 1) + support.intercept

        # Prediction
        if trend == "Uptrend":
            prediction = ("Upward trend confirmed. Price following higher highs and higher lows. "
                          f"Likely continuation toward resistance at {r_val:.4f}.")
        elif trend == "Downtrend":
            prediction = ("Downward trend in force. Lower highs and lower lows pattern. "
                          f"Watch for support test near {s_val:.4f}.")
        else:
            prediction = ("Sideways/consolidation phase. Price oscillating between "
                          f"support ({s_val:.4f}) and resistance ({r_val:.4f}).")

        # Reasoning
        r_desc = f"{resistance.direction.lower()} resistance (slope={resistance.slope:.6f}, R²={resistance.r_squared:.2f})"
        s_desc = f"{support.direction.lower()} support (slope={support.slope:.6f}, R²={support.r_squared:.2f})"

        if abs(resistance.slope) > 0 and abs(support.slope) > 0:
            if np.sign(resistance.slope) == np.sign(support.slope):
                channel_type = "parallel channel"
            else:
                channel_type = "converging wedge"
        else:
            channel_type = "asymmetric channel"

        reasoning = (f"Price compressed in {channel_type}. "
                     f"{r_desc.capitalize()}; {s_desc}. "
                     f"Channel width: {channel_width:.4f}. "
                     f"Breakout probability: {breakout_prob*100:.0f}%. "
                     f"Momentum strength: {self._assess_momentum(trend, resistance.slope - support.slope, last_close, resistance.r_squared, support.r_squared)}.")

        # Signals
        price_vs_mid = (last_close - s_val) / (r_val - s_val + 1e-10)
        if price_vs_mid > 0.7:
            pos_signal = "Price near resistance — potential reversal or breakout zone"
        elif price_vs_mid < 0.3:
            pos_signal = "Price near support — potential bounce or breakdown zone"
        else:
            pos_signal = "Price in mid-channel — directional bias unclear"

        signals = (f"{pos_signal}. "
                   f"Resistance: {'upward-sloping' if resistance.slope > 0 else 'downward-sloping'}. "
                   f"Support: {'upward-sloping' if support.slope > 0 else 'downward-sloping'}. "
                   f"{'High' if breakout_prob > 0.7 else 'Moderate' if breakout_prob > 0.4 else 'Low'} "
                   f"breakout tension.")

        return prediction, reasoning, signals

    def get_trendline_points(self, report: TrendReport,
                              n_points: int = 20) -> Dict:
        """Get x,y coordinates for both trendlines for charting"""
        x = np.arange(n_points, dtype=float)
        resistance_y = report.resistance_line.slope * x + report.resistance_line.intercept
        support_y = report.support_line.slope * x + report.support_line.intercept
        return {
            'x': x.tolist(),
            'resistance': resistance_y.tolist(),
            'support': support_y.tolist()
        }
