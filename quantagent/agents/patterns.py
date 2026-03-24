"""
PatternAgent: Chart Pattern Recognition Module
Replicates the QuantAgent PatternAgent from the paper (Section 3.1.1).
Detects classical patterns: Double Bottom, Head & Shoulders, Triangles, etc.
Uses geometric analysis of price bars (peaks, troughs, consolidations).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.signal import argrelextrema


@dataclass
class PatternMatch:
    name: str
    direction: str          # "Bullish", "Bearish", "Neutral"
    confidence: float       # 0–1
    structure_summary: str
    trend_summary: str
    symmetry_summary: str
    reliability: str        # "High", "Medium", "Low"
    pivot_points: List[Tuple[int, float]] = field(default_factory=list)


# Pattern library as described in paper Appendix C.2
PATTERN_LIBRARY = {
    "Inverse Head and Shoulders": {
        "direction": "Bullish",
        "description": "Three lows with middle being lowest; symmetrical — precedes upward trend",
        "reliability": "High"
    },
    "Double Bottom": {
        "direction": "Bullish",
        "description": "Two similar lows with rebound in between, forming a 'W' shape",
        "reliability": "High"
    },
    "Rounded Bottom": {
        "direction": "Bullish",
        "description": "Gradual decline followed by gradual rise ('U' shape)",
        "reliability": "Medium"
    },
    "Hidden Base": {
        "direction": "Bullish",
        "description": "Horizontal consolidation followed by sudden up-break",
        "reliability": "Medium"
    },
    "Falling Wedge": {
        "direction": "Bullish",
        "description": "Range narrows downward, often resolves upward",
        "reliability": "Medium"
    },
    "Ascending Triangle": {
        "direction": "Bullish",
        "description": "Rising support, flat resistance; breakout usually up",
        "reliability": "High"
    },
    "Bullish Flag": {
        "direction": "Bullish",
        "description": "Sharp rise then brief downward channel before continuation",
        "reliability": "High"
    },
    "V-shaped Reversal (Bullish)": {
        "direction": "Bullish",
        "description": "Sharp decline followed by sharp recovery",
        "reliability": "Medium"
    },
    "Rising Wedge": {
        "direction": "Bearish",
        "description": "Range narrows upward, often resolves downward",
        "reliability": "Medium"
    },
    "Descending Triangle": {
        "direction": "Bearish",
        "description": "Falling resistance, flat support; breakout usually down",
        "reliability": "High"
    },
    "Head and Shoulders": {
        "direction": "Bearish",
        "description": "Three peaks with middle being highest; precedes downward trend",
        "reliability": "High"
    },
    "Double Top": {
        "direction": "Bearish",
        "description": "Two similar highs with pullback, forming 'M' shape",
        "reliability": "High"
    },
    "Bearish Flag": {
        "direction": "Bearish",
        "description": "Sharp drop then brief upward channel before continuation",
        "reliability": "High"
    },
    "Rounded Top": {
        "direction": "Bearish",
        "description": "Gradual peaking followed by gradual decline, arc-shaped",
        "reliability": "Medium"
    },
    "Rectangle": {
        "direction": "Neutral",
        "description": "Sideways range between horizontal support/resistance",
        "reliability": "Medium"
    },
    "Symmetrical Triangle": {
        "direction": "Neutral",
        "description": "Highs and lows converge; breakout after apex",
        "reliability": "Medium"
    },
    "Expanding Triangle": {
        "direction": "Neutral",
        "description": "Highs and lows spread wider, volatile swings",
        "reliability": "Low"
    },
    "Island Reversal": {
        "direction": "Neutral",
        "description": "Two gaps in opposite directions forming an 'island'",
        "reliability": "Medium"
    },
    "No Clear Pattern": {
        "direction": "Neutral",
        "description": "Price action does not match any canonical pattern",
        "reliability": "Low"
    }
}


def find_pivots(prices: np.ndarray, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Find local highs (peaks) and local lows (troughs) in price data"""
    if len(prices) < order * 2 + 1:
        return np.array([]), np.array([])
    peaks = argrelextrema(prices, np.greater_equal, order=order)[0]
    troughs = argrelextrema(prices, np.less_equal, order=order)[0]
    return peaks, troughs


def linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope as used in TrendAgent Algorithm 1"""
    if len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    return num / den if den != 0 else 0.0


def detect_double_bottom(closes: np.ndarray, troughs: np.ndarray,
                          tolerance: float = 0.02) -> float:
    """Detect Double Bottom (W pattern) — confidence score"""
    if len(troughs) < 2:
        return 0.0
    # Take last two troughs
    t1, t2 = troughs[-2], troughs[-1]
    p1, p2 = closes[t1], closes[t2]
    diff = abs(p1 - p2) / max(p1, p2)
    # Check they are similar in price
    if diff > tolerance:
        return 0.0
    # Check there's a peak between them
    mid_prices = closes[t1:t2]
    if len(mid_prices) < 3:
        return 0.0
    mid_high = np.max(mid_prices)
    rebound_ratio = (mid_high - min(p1, p2)) / max(p1, p2)
    if rebound_ratio < 0.005:
        return 0.0
    confidence = max(0.5, 1.0 - diff / tolerance)
    return confidence


def detect_double_top(closes: np.ndarray, peaks: np.ndarray,
                       tolerance: float = 0.02) -> float:
    """Detect Double Top (M pattern) — confidence score"""
    if len(peaks) < 2:
        return 0.0
    t1, t2 = peaks[-2], peaks[-1]
    p1, p2 = closes[t1], closes[t2]
    diff = abs(p1 - p2) / max(p1, p2)
    if diff > tolerance:
        return 0.0
    mid_prices = closes[t1:t2]
    if len(mid_prices) < 3:
        return 0.0
    mid_low = np.min(mid_prices)
    pullback_ratio = (max(p1, p2) - mid_low) / max(p1, p2)
    if pullback_ratio < 0.005:
        return 0.0
    confidence = max(0.5, 1.0 - diff / tolerance)
    return confidence


def detect_head_shoulders(closes: np.ndarray, peaks: np.ndarray) -> float:
    """Detect Head and Shoulders pattern"""
    if len(peaks) < 3:
        return 0.0
    # Need 3 peaks: left shoulder, head, right shoulder
    l, h, r = peaks[-3], peaks[-2], peaks[-1]
    lp, hp, rp = closes[l], closes[h], closes[r]
    # Head must be higher than both shoulders
    if not (hp > lp and hp > rp):
        return 0.0
    # Shoulders roughly equal
    shoulder_diff = abs(lp - rp) / max(lp, rp)
    if shoulder_diff > 0.05:
        return 0.0
    confidence = max(0.4, 1.0 - shoulder_diff / 0.05)
    return confidence


def detect_inverse_head_shoulders(closes: np.ndarray, troughs: np.ndarray) -> float:
    """Detect Inverse Head and Shoulders"""
    if len(troughs) < 3:
        return 0.0
    l, h, r = troughs[-3], troughs[-2], troughs[-1]
    lp, hp, rp = closes[l], closes[h], closes[r]
    if not (hp < lp and hp < rp):
        return 0.0
    shoulder_diff = abs(lp - rp) / max(lp, rp)
    if shoulder_diff > 0.05:
        return 0.0
    confidence = max(0.4, 1.0 - shoulder_diff / 0.05)
    return confidence


def detect_triangle(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     peaks: np.ndarray, troughs: np.ndarray) -> Dict[str, float]:
    """Detect triangle patterns: ascending, descending, symmetrical"""
    results = {}
    if len(peaks) < 2 or len(troughs) < 2:
        return results

    peak_prices = highs[peaks[-min(4, len(peaks)):]]
    trough_prices = lows[troughs[-min(4, len(troughs)):]]

    peak_indices = np.arange(len(peak_prices))
    trough_indices = np.arange(len(trough_prices))

    peak_slope = linear_regression_slope(peak_indices.astype(float),
                                          peak_prices.astype(float))
    trough_slope = linear_regression_slope(trough_indices.astype(float),
                                            trough_prices.astype(float))

    # Normalize slopes
    price_range = np.max(closes) - np.min(closes)
    if price_range == 0:
        return results
    norm_peak = peak_slope / price_range * 10
    norm_trough = trough_slope / price_range * 10

    # Descending Triangle: falling resistance (negative peak_slope), flat support
    if norm_peak < -0.1 and abs(norm_trough) < 0.1:
        results["Descending Triangle"] = 0.75

    # Ascending Triangle: flat resistance, rising support
    elif abs(norm_peak) < 0.1 and norm_trough > 0.1:
        results["Ascending Triangle"] = 0.75

    # Symmetrical Triangle: converging lines
    elif norm_peak < -0.05 and norm_trough > 0.05:
        results["Symmetrical Triangle"] = 0.70

    # Expanding Triangle: diverging
    elif norm_peak > 0.05 and norm_trough < -0.05:
        results["Expanding Triangle"] = 0.55

    # Rising Wedge: both upward but converging
    elif norm_peak > 0.05 and norm_trough > 0.05 and norm_peak < norm_trough:
        results["Rising Wedge"] = 0.60

    # Falling Wedge: both downward but converging
    elif norm_peak < -0.05 and norm_trough < -0.05 and norm_peak > norm_trough:
        results["Falling Wedge"] = 0.60

    return results


def detect_flag(closes: np.ndarray) -> Dict[str, float]:
    """Detect Bull/Bear Flag patterns"""
    results = {}
    if len(closes) < 20:
        return results

    # Check for sharp move in first portion
    first_half = closes[:len(closes)//2]
    second_half = closes[len(closes)//2:]

    first_move = (first_half[-1] - first_half[0]) / first_half[0]
    second_move = (second_half[-1] - second_half[0]) / second_half[0]

    # Bullish flag: sharp rise then slight downward channel
    if first_move > 0.02 and -0.01 < second_move < 0.005:
        results["Bullish Flag"] = 0.65

    # Bearish flag: sharp drop then slight upward channel
    elif first_move < -0.02 and -0.005 < second_move < 0.01:
        results["Bearish Flag"] = 0.65

    return results


def detect_rounded(closes: np.ndarray) -> Dict[str, float]:
    """Detect rounded top/bottom patterns"""
    results = {}
    if len(closes) < 15:
        return results

    n = len(closes)
    mid = n // 2
    first_slope = (closes[mid] - closes[0]) / max(abs(closes[0]), 1)
    second_slope = (closes[-1] - closes[mid]) / max(abs(closes[mid]), 1)

    # Rounded Bottom: down then up
    if first_slope < -0.005 and second_slope > 0.005:
        results["Rounded Bottom"] = 0.55

    # Rounded Top: up then down
    elif first_slope > 0.005 and second_slope < -0.005:
        results["Rounded Top"] = 0.55

    return results


def detect_vshaped(closes: np.ndarray) -> Dict[str, float]:
    """Detect V-shaped reversal"""
    results = {}
    if len(closes) < 10:
        return results

    n = len(closes)
    mid = n // 2
    first_slope = (closes[mid] - closes[0]) / max(abs(closes[0]), 1)
    second_slope = (closes[-1] - closes[mid]) / max(abs(closes[mid]), 1)

    # Sharp decline then sharp recovery
    if first_slope < -0.015 and second_slope > 0.015:
        results["V-shaped Reversal (Bullish)"] = 0.65

    return results


def detect_rectangle(closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray) -> Dict[str, float]:
    """Detect Rectangle / consolidation pattern"""
    results = {}
    if len(closes) < 10:
        return results

    high_range = np.max(highs) - np.min(highs)
    low_range = np.max(lows) - np.min(lows)
    price_range = np.max(closes) - np.min(closes)

    # Tight range = rectangle
    if price_range / np.mean(closes) < 0.02:
        results["Rectangle"] = 0.70

    return results


class PatternAgent:
    """
    PatternAgent: detects chart formations using geometric analysis.
    As described in paper Section 3.1.1:
    - Generates candlestick visualizations
    - Detects visual features: highs/lows, symmetry, reversals
    - Matches against pattern library
    - Produces Structure, Trend, Symmetry summaries
    """

    def __init__(self):
        self.name = "PatternAgent"

    def analyze(self, ohlc_df: pd.DataFrame) -> PatternMatch:
        """
        Analyze OHLC data and return best matching pattern with structured summaries.
        """
        closes = ohlc_df['close'].values.astype(float)
        highs = ohlc_df['high'].values.astype(float)
        lows = ohlc_df['low'].values.astype(float)

        # Find pivot points
        order = max(2, len(closes) // 20)
        peaks, troughs = find_pivots(closes, order=order)

        # Run all pattern detectors
        pattern_scores: Dict[str, float] = {}

        # Double patterns
        db_score = detect_double_bottom(closes, troughs)
        if db_score > 0:
            pattern_scores["Double Bottom"] = db_score

        dt_score = detect_double_top(closes, peaks)
        if dt_score > 0:
            pattern_scores["Double Top"] = dt_score

        hs_score = detect_head_shoulders(closes, peaks)
        if hs_score > 0:
            pattern_scores["Head and Shoulders"] = hs_score

        ihs_score = detect_inverse_head_shoulders(closes, troughs)
        if ihs_score > 0:
            pattern_scores["Inverse Head and Shoulders"] = ihs_score

        # Triangles
        triangles = detect_triangle(closes, highs, lows, peaks, troughs)
        pattern_scores.update(triangles)

        # Flags
        flags = detect_flag(closes)
        pattern_scores.update(flags)

        # Rounded
        rounded = detect_rounded(closes)
        pattern_scores.update(rounded)

        # V-shaped
        vshaped = detect_vshaped(closes)
        pattern_scores.update(vshaped)

        # Rectangle
        rect = detect_rectangle(closes, highs, lows)
        pattern_scores.update(rect)

        # Select best pattern
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            best_score = pattern_scores[best_pattern]
        else:
            best_pattern = "No Clear Pattern"
            best_score = 0.3

        pattern_info = PATTERN_LIBRARY[best_pattern]
        pivot_pts = ([(int(p), float(closes[p])) for p in peaks[-3:]] +
                     [(int(t), float(closes[t])) for t in troughs[-3:]])

        structure, trend, symmetry = self._generate_summaries(
            best_pattern, closes, peaks, troughs, pattern_info['direction']
        )

        return PatternMatch(
            name=best_pattern,
            direction=pattern_info['direction'],
            confidence=best_score,
            structure_summary=structure,
            trend_summary=trend,
            symmetry_summary=symmetry,
            reliability=pattern_info['reliability'],
            pivot_points=sorted(pivot_pts, key=lambda x: x[0])
        )

    def _generate_summaries(self, pattern: str, closes: np.ndarray,
                             peaks: np.ndarray, troughs: np.ndarray,
                             direction: str) -> Tuple[str, str, str]:
        """Generate the three structured summaries as shown in paper Figure 4"""

        # Structure Summary
        if len(peaks) >= 2 and len(troughs) >= 2:
            recent_highs = closes[peaks[-2:]]
            recent_lows = closes[troughs[-2:]]
            if recent_highs[-1] < recent_highs[-2]:
                struct = f"Lower highs sequence ({recent_highs[-2]:.2f} → {recent_highs[-1]:.2f})"
            elif recent_highs[-1] > recent_highs[-2]:
                struct = f"Higher highs sequence ({recent_highs[-2]:.2f} → {recent_highs[-1]:.2f})"
            else:
                struct = f"Equal highs near {recent_highs[-1]:.2f}"

            if recent_lows[-1] > recent_lows[-2]:
                struct += f" with rising support ({recent_lows[-2]:.2f} → {recent_lows[-1]:.2f})"
            elif recent_lows[-1] < recent_lows[-2]:
                struct += f" with falling support ({recent_lows[-2]:.2f} → {recent_lows[-1]:.2f})"
            else:
                struct += f" with flat support near {recent_lows[-1]:.2f}"
        else:
            struct = f"Pattern: {pattern}. Insufficient pivots for detailed structural analysis."

        # Trend Summary
        recent_slope = linear_regression_slope(
            np.arange(min(20, len(closes)), dtype=float),
            closes[-min(20, len(closes)):].astype(float)
        )
        norm_slope = recent_slope / (np.mean(closes) + 1e-10)

        if direction == "Bullish":
            trend = (f"Bullish setup detected. {pattern} typically precedes upward continuation. "
                     f"Current slope: {'rising' if norm_slope > 0 else 'declining'}. "
                     f"Breakout confirmation recommended before entry.")
        elif direction == "Bearish":
            trend = (f"Bearish setup detected. {pattern} signals potential downward move. "
                     f"Current slope: {'declining' if norm_slope < 0 else 'rising'}. "
                     f"Watch for breakdown below support.")
        else:
            trend = (f"Neutral/consolidation pattern. {pattern} indicates indecision. "
                     f"Await breakout direction before committing to a side.")

        # Symmetry Summary
        price_range = np.max(closes) - np.min(closes)
        price_center = (np.max(closes) + np.min(closes)) / 2
        if "Triangle" in pattern:
            symmetry = (f"Triangular convergence detected. Price range compressing "
                        f"(span: {price_range:.4f}). Apex forming — expect breakout.")
        elif "Double" in pattern or "Head" in pattern:
            symmetry = (f"Bilateral symmetry present. Two structural points "
                        f"near {price_center:.4f}. Pattern completion increases reversal probability.")
        elif "Wedge" in pattern:
            symmetry = (f"Wedge convergence: both boundaries narrowing. "
                        f"Range: {price_range:.4f}. Compression builds directional pressure.")
        elif "Flag" in pattern:
            symmetry = (f"Flag channel: parallel consolidation after impulse move. "
                        f"Continuation pattern — watch pole height for target projection.")
        else:
            symmetry = (f"Price oscillating around center {price_center:.4f}. "
                        f"Range: {price_range:.4f}. No strong geometric symmetry detected.")

        return struct, trend, symmetry
