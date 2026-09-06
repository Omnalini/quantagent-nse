"""
RiskAgent + DecisionAgent Module
Replicates paper Section 3.1.3 (RiskAgent) and Figure 3 (DecisionAgent).

RiskAgent:
- Integrates Indicator, Pattern, Trend signals into unified risk-reward framework
- Sets stop-loss ρ = 0.0005 (0.05%)
- Computes take-profit R = r * ρ where r ∈ [1.2, 1.8]
- Produces radar chart dimensions: Momentum Strength, Volatility Risk,
  S/R Strength, Breakout Likelihood, Trend Certainty, Pattern Reliability

DecisionAgent:
- Produces LONG or SHORT (no HOLD)
- Forecasts next 3 candlesticks
- Outputs: direction, justification, trade setup, post-trade reflection
- Requires confirmed, aligned signals across all three agent reports
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from .indicators import IndicatorReport
from .patterns import PatternMatch
from .trend import TrendReport


# Fixed stop-loss from paper Section 3.1.3
STOP_LOSS_RHO = 0.0005  # 0.05%
RR_RATIO_MIN = 1.2
RR_RATIO_MAX = 1.8


@dataclass
class RadarDimensions:
    """Six radar chart dimensions from paper Figure 2 / RiskAgent"""
    momentum_strength: float       # Speed and persistence of price movement (0–1)
    volatility_risk: float         # Magnitude of price fluctuations (0–1)
    sr_strength: float             # Integrity of key price zones (0–1)
    breakout_likelihood: float     # Probability of escaping consolidation (0–1)
    trend_certainty: float         # Clarity and consistency of directional bias (0–1)
    pattern_reliability: float     # Validity and completion of chart formation (0–1)


@dataclass
class RiskAssessment:
    radar: RadarDimensions
    stop_loss: float               # Price level for stop-loss
    take_profit: float             # Price level for take-profit
    entry_price: float
    risk_reward_ratio: float
    direction: str                 # "LONG" or "SHORT"
    risk_zone_width: float
    reward_zone_width: float
    overall_risk_score: float      # 0–1 (1 = highest risk)


@dataclass
class TradeDecision:
    direction: str                         # "LONG" or "SHORT"
    forecast_horizon: str                  # e.g., "Predicting next 3 candlesticks"
    justification: str
    trade_setup: str
    reflection: str
    risk_reward_ratio: float
    stop_loss: float
    take_profit: float
    entry_price: float
    confidence: float
    signal_alignment: Dict[str, str]       # Which signals aligned
    radar: RadarDimensions


def compute_volatility(closes: np.ndarray, window: int = 20) -> float:
    """Compute normalized volatility (ATR-like)"""
    if len(closes) < 2:
        return 0.5
    returns = np.diff(np.log(np.maximum(closes, 1e-10)))
    vol = np.std(returns[-min(window, len(returns)):])
    # Normalize to 0–1 range (cap at 5% daily vol)
    return float(min(1.0, vol / 0.05))


def compute_sr_strength(trend_report: TrendReport) -> float:
    """Assess support/resistance zone strength"""
    r2_avg = (trend_report.resistance_line.r_squared +
               trend_report.support_line.r_squared) / 2
    return float(np.clip(r2_avg, 0, 1))


def compute_momentum_strength(indicator_report: IndicatorReport) -> float:
    """Derive momentum strength from indicator signals"""
    score = 0.5
    # RSI contribution
    rsi_val = indicator_report.rsi.value
    if rsi_val > 60 or rsi_val < 40:
        score += 0.1
    # MACD contribution
    if indicator_report.macd.signal != "Neutral":
        score += 0.15
    # ROC contribution
    roc_val = abs(indicator_report.roc.value)
    score += min(0.15, roc_val * 0.1)
    return float(np.clip(score, 0, 1))


def compute_risk_reward_ratio(indicator_report: IndicatorReport,
                               pattern_match: PatternMatch,
                               trend_report: TrendReport) -> float:
    """
    Determine LLM-equivalent risk-reward ratio r ∈ [1.2, 1.8]
    Based on signal quality and alignment
    """
    # Count aligned signals
    signals = [
        indicator_report.overall_signal,
        pattern_match.direction,
        trend_report.trend.replace("Uptrend", "Bullish")
               .replace("Downtrend", "Bearish")
               .replace("Sideways", "Neutral")
    ]

    # Determine dominant direction
    bull = sum(1 for s in signals if s == "Bullish")
    bear = sum(1 for s in signals if s == "Bearish")
    aligned = max(bull, bear)

    # More alignment = higher ratio (more confidence = more reward vs risk)
    base_ratio = RR_RATIO_MIN
    alignment_bonus = (aligned / len(signals)) * (RR_RATIO_MAX - RR_RATIO_MIN)
    pattern_bonus = (pattern_match.confidence - 0.5) * 0.2 if pattern_match.confidence > 0.5 else 0

    rr = base_ratio + alignment_bonus + pattern_bonus
    return float(np.clip(rr, RR_RATIO_MIN, RR_RATIO_MAX))


class RiskAgent:
    """
    RiskAgent: translates technical insights into risk-aware trade boundaries.
    From paper Section 3.1.3:
    - ρ = 0.0005 (fixed stop-loss)
    - R = r * ρ (take-profit, r predicted by LLM/system)
    - Produces radar chart of 6 dimensions
    """

    def __init__(self, rho: float = STOP_LOSS_RHO):
        """
        rho: fractional stop-loss distance from entry. Defaults to the paper's
        0.0005 (0.05%), which is what every published result in this repo was
        produced with — override it only for experiments.
        """
        self.name = "RiskAgent"
        self.rho = rho

    def analyze(self, ohlc_df: pd.DataFrame,
                indicator_report: IndicatorReport,
                pattern_match: PatternMatch,
                trend_report: TrendReport) -> RiskAssessment:

        closes = ohlc_df['close'].values.astype(float)
        entry_price = float(closes[-1])

        # Determine direction from aligned signals
        direction = self._determine_direction(indicator_report, pattern_match, trend_report)

        # Compute radar dimensions
        volatility = compute_volatility(closes)
        momentum = compute_momentum_strength(indicator_report)
        sr_str = compute_sr_strength(trend_report)
        breakout_prob = trend_report.breakout_probability
        pattern_rel = pattern_match.confidence

        # Trend certainty from trendline R²
        trend_cert = (trend_report.resistance_line.r_squared +
                      trend_report.support_line.r_squared) / 2

        radar = RadarDimensions(
            momentum_strength=float(np.clip(momentum, 0, 1)),
            volatility_risk=float(np.clip(volatility, 0, 1)),
            sr_strength=float(np.clip(sr_str, 0, 1)),
            breakout_likelihood=float(np.clip(breakout_prob, 0, 1)),
            trend_certainty=float(np.clip(trend_cert, 0, 1)),
            pattern_reliability=float(np.clip(pattern_rel, 0, 1))
        )

        # Risk-reward ratio
        rr_ratio = compute_risk_reward_ratio(indicator_report, pattern_match, trend_report)

        # Stop-loss and take-profit levels
        if direction == "LONG":
            stop_loss = entry_price * (1 - self.rho)
            take_profit = entry_price * (1 + rr_ratio * self.rho)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.rho)
            take_profit = entry_price * (1 - rr_ratio * self.rho)

        risk_zone = abs(entry_price - stop_loss)
        reward_zone = abs(take_profit - entry_price)
        overall_risk = float(np.clip(volatility * 0.4 + (1 - sr_str) * 0.3 +
                                     (1 - trend_cert) * 0.3, 0, 1))

        return RiskAssessment(
            radar=radar,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=entry_price,
            risk_reward_ratio=rr_ratio,
            direction=direction,
            risk_zone_width=risk_zone,
            reward_zone_width=reward_zone,
            overall_risk_score=overall_risk
        )

    def _determine_direction(self, indicator: IndicatorReport,
                              pattern: PatternMatch,
                              trend: TrendReport) -> str:
        """Determine LONG or SHORT from signal alignment"""
        bull_score = 0
        bear_score = 0

        # Indicator signal
        if indicator.overall_signal == "Bullish":
            bull_score += 1.5
        elif indicator.overall_signal == "Bearish":
            bear_score += 1.5

        # Pattern signal
        if pattern.direction == "Bullish":
            bull_score += pattern.confidence
        elif pattern.direction == "Bearish":
            bear_score += pattern.confidence

        # Trend signal
        if trend.trend == "Uptrend":
            bull_score += 1.2
        elif trend.trend == "Downtrend":
            bear_score += 1.2

        # Individual indicator sub-signals
        for sig in [indicator.rsi, indicator.macd, indicator.roc,
                    indicator.stoch, indicator.willr]:
            if sig.signal == "Bullish":
                bull_score += 0.3
            elif sig.signal == "Bearish":
                bear_score += 0.3

        return "LONG" if bull_score >= bear_score else "SHORT"


class DecisionAgent:
    """
    DecisionAgent: integrates upstream agent outputs into final trade decision.
    From paper Section 3.1.3 and Figure 3:
    - Outputs LONG or SHORT (HOLD prohibited)
    - Forecasts next 3 candlesticks
    - Requires majority alignment + confirmation
    - Produces: direction, justification, trade setup, post-trade reflection
    """

    def __init__(self, timeframe: str = "15m"):
        self.name = "DecisionAgent"
        self.timeframe = timeframe

    def decide(self, ohlc_df: pd.DataFrame,
               indicator_report: IndicatorReport,
               pattern_match: PatternMatch,
               trend_report: TrendReport,
               risk_assessment: RiskAssessment) -> TradeDecision:
        """
        Generate final trade decision integrating all upstream signals.
        """
        closes = ohlc_df['close'].values.astype(float)
        entry = float(closes[-1])
        direction = risk_assessment.direction

        # Build signal alignment map
        signal_alignment = {
            "Indicator": indicator_report.overall_signal,
            "Pattern": pattern_match.direction,
            "Trend": (trend_report.trend.replace("Uptrend", "Bullish")
                                         .replace("Downtrend", "Bearish")
                                         .replace("Sideways", "Neutral")),
            "RSI": indicator_report.rsi.signal,
            "MACD": indicator_report.macd.signal,
        }

        aligned_count = sum(1 for v in signal_alignment.values()
                            if (direction == "LONG" and v == "Bullish") or
                               (direction == "SHORT" and v == "Bearish"))

        confidence = min(0.95, aligned_count / len(signal_alignment) + 0.3)

        # Forecast horizon
        tf_map = {"1m": "1 min", "5m": "5 min", "15m": "15 min",
                  "30m": "30 min", "1h": "1 hour", "4h": "4 hours", "1d": "1 day"}
        tf_label = tf_map.get(self.timeframe, self.timeframe)
        horizon = f"Predicting next 3 candlesticks (~{tf_label} each)"

        justification = self._build_justification(
            direction, indicator_report, pattern_match, trend_report,
            signal_alignment, aligned_count
        )

        trade_setup = self._build_trade_setup(
            direction, entry, risk_assessment
        )

        reflection = self._build_reflection(
            direction, indicator_report, pattern_match, trend_report, confidence
        )

        return TradeDecision(
            direction=direction,
            forecast_horizon=horizon,
            justification=justification,
            trade_setup=trade_setup,
            reflection=reflection,
            risk_reward_ratio=risk_assessment.risk_reward_ratio,
            stop_loss=risk_assessment.stop_loss,
            take_profit=risk_assessment.take_profit,
            entry_price=entry,
            confidence=confidence,
            signal_alignment=signal_alignment,
            radar=risk_assessment.radar
        )

    def _build_justification(self, direction: str,
                              indicator: IndicatorReport,
                              pattern: PatternMatch,
                              trend: TrendReport,
                              alignment: Dict[str, str],
                              aligned_count: int) -> str:
        """Build concise confirmed reasoning as per paper DecisionAgent prompt"""
        dir_word = "bullish" if direction == "LONG" else "bearish"
        parts = []

        # Indicator contribution
        parts.append(f"Technical Indicator: {indicator.summary}")

        # Pattern contribution
        parts.append(f"Pattern: {pattern.name} detected ({pattern.direction}, "
                     f"confidence {pattern.confidence:.0%}). {pattern.trend_summary}")

        # Trend contribution
        parts.append(f"Trend: {trend.trend}. {trend.reasoning}")

        # Alignment summary
        parts.append(f"Signal alignment: {aligned_count}/{len(alignment)} signals "
                     f"confirm {dir_word} bias. "
                     f"Decision: {direction} — more defensible given current market structure.")

        return " | ".join(parts)

    def _build_trade_setup(self, direction: str, entry: float,
                            risk: RiskAssessment) -> str:
        """Build trade setup with entry, stop-loss, take-profit"""
        # Derived from the levels themselves so the label stays correct if a
        # RiskAgent is constructed with a non-default rho.
        sl_pct = abs(entry - risk.stop_loss) / entry * 100 if entry else 0.0
        return (
            f"Direction: {direction} | "
            f"Entry: {entry:.4f} | "
            f"Stop-Loss: {risk.stop_loss:.4f} ({sl_pct:.2f}% from entry) | "
            f"Take-Profit: {risk.take_profit:.4f} "
            f"(R:R = 1:{risk.risk_reward_ratio:.2f}) | "
            f"Risk Zone: {risk.risk_zone_width:.4f} | "
            f"Reward Zone: {risk.reward_zone_width:.4f}"
        )

    def _build_reflection(self, direction: str,
                           indicator: IndicatorReport,
                           pattern: PatternMatch,
                           trend: TrendReport,
                           confidence: float) -> str:
        """Post-trade reflection — what to watch for"""
        if confidence > 0.7:
            conf_text = "High-confidence setup"
        elif confidence > 0.5:
            conf_text = "Moderate-confidence setup"
        else:
            conf_text = "Low-confidence setup — exercise caution"

        if direction == "LONG":
            watch = ("Watch for: momentum stall or RSI overbought reversal, "
                     "price rejection at resistance, bearish MACD crossover as invalidation.")
        else:
            watch = ("Watch for: oversold bounce or RSI recovery, "
                     "price holding support, bullish reversal pattern as invalidation.")

        return (f"{conf_text} (confidence: {confidence:.0%}). "
                f"Predominant signals: Indicator={indicator.overall_signal}, "
                f"Pattern={pattern.direction}, Trend={trend.trend}. {watch}")
