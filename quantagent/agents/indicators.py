"""
IndicatorAgent: Technical Indicators Module
Replicates the QuantAgent IndicatorAgent from the paper.
Computes RSI, MACD, ROC, Stochastic Oscillator, Williams %R
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class IndicatorSignal:
    name: str
    value: float
    signal: str          # "Bullish", "Bearish", "Neutral"
    description: str
    raw_values: List[float] = None


@dataclass
class IndicatorReport:
    rsi: IndicatorSignal
    macd: IndicatorSignal
    roc: IndicatorSignal
    stoch: IndicatorSignal
    willr: IndicatorSignal
    overall_signal: str
    confidence: float
    summary: str


def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average as described in paper Appendix G"""
    alpha = 2.0 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI: captures momentum and flags overbought/oversold zones (Wilder, 1978)
    RSI > 70: Overbought (Bearish signal)
    RSI < 30: Oversold (Bullish signal)
    30 < RSI < 70: Neutral
    """
    if len(closes) < period + 1:
        return np.array([50.0])
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = np.zeros(len(deltas))
    avg_loss = np.zeros(len(deltas))
    
    # Initial average
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])
    
    # Wilder smoothing
    for i in range(period, len(deltas)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    
    rs = np.where(avg_loss[period-1:] != 0,
                  avg_gain[period-1:] / avg_loss[period-1:],
                  100.0)
    rsi = 100 - (100 / (1 + rs))
    return np.clip(rsi, 0, 100)


def compute_macd(closes: np.ndarray,
                 fast: int = 12,
                 slow: int = 26,
                 signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD: tracks convergence/divergence between short/long-term price trends (Appel, 2005)
    As per paper Appendix G: Mt = E_fast - E_slow, St = EMA(Mt, 9)
    Returns: macd_line, signal_line, histogram
    """
    if len(closes) < slow + signal_period:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_roc(closes: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Rate of Change: measures speed of price changes (Murphy, 1999)
    ROC = ((Close - Close[n]) / Close[n]) * 100
    ROC > 0: Bullish, ROC < 0: Bearish
    """
    if len(closes) < period + 1:
        return np.array([0.0])
    roc = ((closes[period:] - closes[:-period]) / closes[:-period]) * 100
    return roc


def compute_stochastic(highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray, k_period: int = 14,
                        d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator: identifies turning points based on recent highs/lows
    %K > 80: Overbought, %K < 20: Oversold
    """
    if len(closes) < k_period:
        return np.array([50.0]), np.array([50.0])
    
    k_values = []
    for i in range(k_period - 1, len(closes)):
        window_high = np.max(highs[i - k_period + 1:i + 1])
        window_low = np.min(lows[i - k_period + 1:i + 1])
        denom = window_high - window_low
        if denom == 0:
            k_values.append(50.0)
        else:
            k = ((closes[i] - window_low) / denom) * 100
            k_values.append(np.clip(k, 0, 100))
    
    k_arr = np.array(k_values)
    # %D is SMA of %K
    if len(k_arr) >= d_period:
        d_arr = np.convolve(k_arr, np.ones(d_period) / d_period, mode='valid')
        # Pad to same length
        pad = len(k_arr) - len(d_arr)
        d_arr = np.concatenate([np.full(pad, d_arr[0]), d_arr])
    else:
        d_arr = k_arr.copy()
    
    return k_arr, d_arr


def compute_williams_r(highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Williams %R: detects price drops from recent peaks to signal possible reversals (Williams, 2011)
    %R > -20: Overbought (Bearish), %R < -80: Oversold (Bullish)
    """
    if len(closes) < period:
        return np.array([-50.0])
    
    willr = []
    for i in range(period - 1, len(closes)):
        highest_high = np.max(highs[i - period + 1:i + 1])
        lowest_low = np.min(lows[i - period + 1:i + 1])
        denom = highest_high - lowest_low
        if denom == 0:
            willr.append(-50.0)
        else:
            wr = ((highest_high - closes[i]) / denom) * -100
            willr.append(np.clip(wr, -100, 0))
    
    return np.array(willr)


def interpret_rsi(rsi_val: float) -> IndicatorSignal:
    """Interpret RSI signal as per paper Figure 1"""
    if rsi_val > 70:
        signal = "Bearish"
        desc = f"RSI at {rsi_val:.1f} — overbought zone. Potential reversal downward."
    elif rsi_val < 30:
        signal = "Bullish"
        desc = f"RSI at {rsi_val:.1f} — oversold zone. Potential reversal upward."
    else:
        signal = "Neutral"
        desc = f"RSI at {rsi_val:.1f} — neutral zone (30–70). No extreme momentum."
    return IndicatorSignal("RSI", rsi_val, signal, desc)


def interpret_macd(macd_val: float, signal_val: float, hist_val: float) -> IndicatorSignal:
    """Interpret MACD crossover signals as per paper"""
    if macd_val > signal_val and hist_val > 0:
        signal = "Bullish"
        desc = f"MACD ({macd_val:.4f}) above signal ({signal_val:.4f}). Bullish crossover — upward momentum."
    elif macd_val < signal_val and hist_val < 0:
        signal = "Bearish"
        desc = f"MACD ({macd_val:.4f}) below signal ({signal_val:.4f}). Bearish crossover — downward momentum."
    else:
        signal = "Neutral"
        desc = f"MACD ({macd_val:.4f}) near signal ({signal_val:.4f}). Momentum consolidating."
    return IndicatorSignal("MACD", macd_val, signal, desc)


def interpret_roc(roc_val: float) -> IndicatorSignal:
    """Interpret Rate of Change"""
    if roc_val > 0.5:
        signal = "Bullish"
        desc = f"ROC at {roc_val:.2f}% — positive price momentum."
    elif roc_val < -0.5:
        signal = "Bearish"
        desc = f"ROC at {roc_val:.2f}% — negative price momentum."
    else:
        signal = "Neutral"
        desc = f"ROC at {roc_val:.2f}% — minimal momentum shift."
    return IndicatorSignal("ROC", roc_val, signal, desc)


def interpret_stochastic(k_val: float, d_val: float) -> IndicatorSignal:
    """Interpret Stochastic Oscillator"""
    if k_val > 80:
        signal = "Bearish"
        desc = f"%K={k_val:.1f}, %D={d_val:.1f} — overbought. Watch for reversal."
    elif k_val < 20:
        signal = "Bullish"
        desc = f"%K={k_val:.1f}, %D={d_val:.1f} — oversold. Watch for bounce."
    else:
        signal = "Neutral"
        desc = f"%K={k_val:.1f}, %D={d_val:.1f} — neutral range."
    return IndicatorSignal("STOCH", k_val, signal, desc)


def interpret_willr(willr_val: float) -> IndicatorSignal:
    """Interpret Williams %R as per paper"""
    if willr_val > -20:
        signal = "Bearish"
        desc = f"Williams %R at {willr_val:.1f} — near peak, potential reversal down."
    elif willr_val < -80:
        signal = "Bullish"
        desc = f"Williams %R at {willr_val:.1f} — near trough, potential reversal up."
    else:
        signal = "Neutral"
        desc = f"Williams %R at {willr_val:.1f} — neutral zone."
    return IndicatorSignal("WILLR", willr_val, signal, desc)


class IndicatorAgent:
    """
    IndicatorAgent: transforms raw OHLC sequences into structured quantitative signals.
    As described in paper Section 3.1:
    - Maps (O,H,L,C) -> S (signal space)
    - Uses RSI, MACD, ROC, STOCH, WILLR
    """

    def __init__(self):
        self.name = "IndicatorAgent"

    def analyze(self, ohlc_df: pd.DataFrame) -> IndicatorReport:
        """
        Main analysis function. Takes OHLC DataFrame with columns:
        ['open', 'high', 'low', 'close', 'volume']
        Returns structured IndicatorReport
        """
        closes = ohlc_df['close'].values.astype(float)
        highs = ohlc_df['high'].values.astype(float)
        lows = ohlc_df['low'].values.astype(float)

        # Compute all indicators
        rsi_series = compute_rsi(closes)
        rsi_val = float(rsi_series[-1]) if len(rsi_series) > 0 else 50.0

        macd_line, signal_line, histogram = compute_macd(closes)
        macd_val = float(macd_line[-1])
        signal_val = float(signal_line[-1])
        hist_val = float(histogram[-1])

        roc_series = compute_roc(closes)
        roc_val = float(roc_series[-1]) if len(roc_series) > 0 else 0.0

        k_series, d_series = compute_stochastic(highs, lows, closes)
        k_val = float(k_series[-1]) if len(k_series) > 0 else 50.0
        d_val = float(d_series[-1]) if len(d_series) > 0 else 50.0

        willr_series = compute_williams_r(highs, lows, closes)
        willr_val = float(willr_series[-1]) if len(willr_series) > 0 else -50.0

        # Build signals
        rsi_sig = interpret_rsi(rsi_val)
        macd_sig = interpret_macd(macd_val, signal_val, hist_val)
        roc_sig = interpret_roc(roc_val)
        stoch_sig = interpret_stochastic(k_val, d_val)
        willr_sig = interpret_willr(willr_val)

        # Store raw series for charting
        rsi_sig.raw_values = rsi_series.tolist()[-50:]
        macd_sig.raw_values = {
            'macd': macd_line.tolist()[-50:],
            'signal': signal_line.tolist()[-50:],
            'histogram': histogram.tolist()[-50:]
        }

        # Aggregate signal
        signals = [rsi_sig.signal, macd_sig.signal, roc_sig.signal,
                   stoch_sig.signal, willr_sig.signal]
        bull_count = signals.count("Bullish") 
        bear_count = signals.count("Bearish")

        if bull_count >= 3:
            overall = "Bullish"
            confidence = bull_count / 5.0
        elif bear_count >= 3:
            overall = "Bearish"
            confidence = bear_count / 5.0
        else:
            overall = "Neutral"
            confidence = 0.4

        summary = self._generate_summary(rsi_sig, macd_sig, roc_sig,
                                          stoch_sig, willr_sig, overall)

        return IndicatorReport(
            rsi=rsi_sig,
            macd=macd_sig,
            roc=roc_sig,
            stoch=stoch_sig,
            willr=willr_sig,
            overall_signal=overall,
            confidence=confidence,
            summary=summary
        )

    def _generate_summary(self, rsi, macd, roc, stoch, willr, overall) -> str:
        parts = []
        if macd.signal != "Neutral":
            parts.append(f"MACD shows {macd.signal.lower()} crossover")
        if rsi.signal == "Bearish":
            parts.append("RSI in overbought territory")
        elif rsi.signal == "Bullish":
            parts.append("RSI in oversold territory")
        if stoch.signal != "Neutral":
            parts.append(f"Stochastic {stoch.signal.lower()}")
        if willr.signal != "Neutral":
            parts.append(f"Williams %R signals {willr.signal.lower()}")

        base = f"Overall: {overall} sentiment. "
        if parts:
            base += "Key signals: " + "; ".join(parts) + "."
        else:
            base += "Mixed signals — exercise caution."
        return base
