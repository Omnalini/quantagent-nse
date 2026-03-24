"""
QuantAgent Orchestrator
Coordinates all four specialized agents as described in the paper:
IndicatorAgent → PatternAgent → TrendAgent → RiskAgent → DecisionAgent
"""

import numpy as np
import pandas as pd
import json
import time
from dataclasses import asdict
from typing import Optional, Dict, Any
from .indicators import IndicatorAgent
from .patterns import PatternAgent
from .trend import TrendAgent
from .risk_decision import RiskAgent, DecisionAgent, TradeDecision


class QuantAgent:
    """
    Multi-agent orchestrator for high-frequency trading analysis.
    Implements the full QuantAgent pipeline from the paper.
    """

    def __init__(self, timeframe: str = "15m", trend_window: int = 20):
        self.timeframe = timeframe
        self.indicator_agent = IndicatorAgent()
        self.pattern_agent = PatternAgent()
        self.trend_agent = TrendAgent(window=trend_window)
        self.risk_agent = RiskAgent()
        self.decision_agent = DecisionAgent(timeframe=timeframe)

    def run(self, ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Full pipeline: run all four agents and return complete analysis.
        Returns serializable dict.
        """
        if len(ohlc_df) < 30:
            return {"error": "Insufficient data (need at least 30 bars)"}

        t0 = time.time()

        # Step 1: IndicatorAgent
        indicator_report = self.indicator_agent.analyze(ohlc_df)

        # Step 2: PatternAgent
        pattern_match = self.pattern_agent.analyze(ohlc_df)

        # Step 3: TrendAgent
        trend_report = self.trend_agent.analyze(ohlc_df)

        # Step 4: RiskAgent
        risk_assessment = self.risk_agent.analyze(
            ohlc_df, indicator_report, pattern_match, trend_report
        )

        # Step 5: DecisionAgent
        decision = self.decision_agent.decide(
            ohlc_df, indicator_report, pattern_match, trend_report, risk_assessment
        )

        elapsed = time.time() - t0

        # Build serializable result
        return self._serialize_result(
            decision, indicator_report, pattern_match,
            trend_report, risk_assessment, ohlc_df, elapsed
        )

    def _serialize_result(self, decision: TradeDecision,
                           ind, pat, tre, risk, ohlc_df, elapsed) -> Dict:
        closes = ohlc_df['close'].values.astype(float)
        highs = ohlc_df['high'].values.astype(float)
        lows = ohlc_df['low'].values.astype(float)
        opens = ohlc_df['open'].values.astype(float)

        # Trendline points for chart
        N = min(20, len(closes))
        x = np.arange(N, dtype=float)
        resist_y = (tre.resistance_line.slope * x + tre.resistance_line.intercept).tolist()
        support_y = (tre.support_line.slope * x + tre.support_line.intercept).tolist()

        result = {
            "timestamp": time.time(),
            "timeframe": self.timeframe,
            "elapsed_ms": round(elapsed * 1000, 1),

            # Final Decision
            "decision": {
                "direction": decision.direction,
                "forecast_horizon": decision.forecast_horizon,
                "justification": decision.justification,
                "trade_setup": decision.trade_setup,
                "reflection": decision.reflection,
                "risk_reward_ratio": round(decision.risk_reward_ratio, 3),
                "stop_loss": round(decision.stop_loss, 6),
                "take_profit": round(decision.take_profit, 6),
                "entry_price": round(decision.entry_price, 6),
                "confidence": round(decision.confidence, 3),
                "signal_alignment": decision.signal_alignment,
            },

            # IndicatorAgent
            "indicators": {
                "rsi": {
                    "value": round(ind.rsi.value, 2),
                    "signal": ind.rsi.signal,
                    "description": ind.rsi.description,
                    "series": ind.rsi.raw_values or []
                },
                "macd": {
                    "value": round(ind.macd.value, 6),
                    "signal": ind.macd.signal,
                    "description": ind.macd.description,
                    "series": ind.macd.raw_values or {}
                },
                "roc": {
                    "value": round(ind.roc.value, 4),
                    "signal": ind.roc.signal,
                    "description": ind.roc.description,
                },
                "stoch": {
                    "value": round(ind.stoch.value, 2),
                    "signal": ind.stoch.signal,
                    "description": ind.stoch.description,
                },
                "willr": {
                    "value": round(ind.willr.value, 2),
                    "signal": ind.willr.signal,
                    "description": ind.willr.description,
                },
                "overall": ind.overall_signal,
                "confidence": round(ind.confidence, 3),
                "summary": ind.summary,
            },

            # PatternAgent
            "pattern": {
                "name": pat.name,
                "direction": pat.direction,
                "confidence": round(pat.confidence, 3),
                "reliability": pat.reliability,
                "structure_summary": pat.structure_summary,
                "trend_summary": pat.trend_summary,
                "symmetry_summary": pat.symmetry_summary,
                "pivot_points": pat.pivot_points,
            },

            # TrendAgent
            "trend": {
                "classification": tre.trend,
                "kappa": round(tre.kappa, 8),
                "momentum_strength": tre.momentum_strength,
                "prediction": tre.prediction,
                "reasoning": tre.reasoning,
                "signals": tre.signals,
                "channel_width": round(tre.channel_width, 6),
                "breakout_probability": round(tre.breakout_probability, 3),
                "resistance": {
                    "slope": round(tre.resistance_line.slope, 8),
                    "intercept": round(tre.resistance_line.intercept, 4),
                    "direction": tre.resistance_line.direction,
                    "strength": tre.resistance_line.strength,
                    "r_squared": round(tre.resistance_line.r_squared, 3),
                    "series": resist_y,
                },
                "support": {
                    "slope": round(tre.support_line.slope, 8),
                    "intercept": round(tre.support_line.intercept, 4),
                    "direction": tre.support_line.direction,
                    "strength": tre.support_line.strength,
                    "r_squared": round(tre.support_line.r_squared, 3),
                    "series": support_y,
                },
                "pivot_highs": tre.pivot_highs,
                "pivot_lows": tre.pivot_lows,
            },

            # RiskAgent
            "risk": {
                "direction": risk.direction,
                "entry_price": round(risk.entry_price, 6),
                "stop_loss": round(risk.stop_loss, 6),
                "take_profit": round(risk.take_profit, 6),
                "risk_reward_ratio": round(risk.risk_reward_ratio, 3),
                "risk_zone_width": round(risk.risk_zone_width, 6),
                "reward_zone_width": round(risk.reward_zone_width, 6),
                "overall_risk_score": round(risk.overall_risk_score, 3),
                "radar": {
                    "momentum_strength": round(risk.radar.momentum_strength, 3),
                    "volatility_risk": round(risk.radar.volatility_risk, 3),
                    "sr_strength": round(risk.radar.sr_strength, 3),
                    "breakout_likelihood": round(risk.radar.breakout_likelihood, 3),
                    "trend_certainty": round(risk.radar.trend_certainty, 3),
                    "pattern_reliability": round(risk.radar.pattern_reliability, 3),
                },
            },

            # OHLC data for charting (last 100 bars)
            "ohlc": {
                "open": opens[-100:].tolist(),
                "high": highs[-100:].tolist(),
                "low": lows[-100:].tolist(),
                "close": closes[-100:].tolist(),
                "n": min(100, len(closes))
            }
        }

        return result
