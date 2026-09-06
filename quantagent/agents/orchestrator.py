"""
QuantAgent Orchestrator — Nifty 50 Edition
Runs the five-agent pipeline and optionally enriches results with Claude LLM.

Pipeline:
  IndicatorAgent → PatternAgent → TrendAgent → RiskAgent → DecisionAgent
                                              ↑ (LLM pattern, if key provided)
                                                                    ↑ (LLM decision, if key provided)
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, Dict, Any

from .indicators import IndicatorAgent
from .patterns import PatternAgent
from .trend import TrendAgent
from .risk_decision import RiskAgent, DecisionAgent, TradeDecision
from .llm_client import LLMClient


class QuantAgent:
    """
    Multi-agent orchestrator for Nifty 50 / NSE high-frequency analysis.

    Args:
        timeframe:   Bar interval string, e.g. "15m".
        trend_window: Number of bars used by TrendAgent for OLS fitting.
        api_key:     Anthropic API key (optional). When present, enables
                     LLM-enhanced pattern recognition and decision synthesis.
        symbol:      NSE ticker (e.g. "RELIANCE") — passed to LLM prompts.
        horizon_min: Forecast window in minutes (3 × bar_minutes by default).
    """

    def __init__(
        self,
        timeframe: str = "15m",
        trend_window: int = 20,
        api_key: Optional[str] = None,
        symbol: str = "NIFTY50",
        horizon_min: int = 45,
    ):
        self.timeframe   = timeframe
        self.symbol      = symbol
        self.horizon_min = horizon_min

        self.indicator_agent = IndicatorAgent()
        self.pattern_agent   = PatternAgent()
        self.trend_agent     = TrendAgent(window=trend_window)
        self.risk_agent      = RiskAgent()
        self.decision_agent  = DecisionAgent(timeframe=timeframe)

        self.llm = LLMClient(api_key=api_key)

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, ohlc_df: pd.DataFrame, require_llm: bool = True) -> Dict[str, Any]:
        """
        Run all agents on the supplied OHLC DataFrame and return a
        fully-serializable result dict.

        Minimum required: 30 bars.
        require_llm: If True (default), returns an error when no API key is set.
                     Pass False for backtest/offline use — algorithmic signals only.
        """
        if len(ohlc_df) < 30:
            return {"error": "Insufficient data (need ≥30 bars)"}

        if require_llm and not self.llm.available:
            return {
                "error": "no_api_key",
                "llm_required": True,
                "message": "An API key is required to run analysis. "
                           "Enter a Gemini key (AIza…) or Claude key (sk-ant-…) in the sidebar.",
            }

        t0 = time.time()

        # ── Step 1: Indicators ────────────────────────────────────────
        indicator_report = self.indicator_agent.analyze(ohlc_df)

        # ── Step 2: Patterns (algorithmic) ────────────────────────────
        pattern_match = self.pattern_agent.analyze(ohlc_df)

        # ── Step 3: Trend ─────────────────────────────────────────────
        trend_report = self.trend_agent.analyze(ohlc_df)

        # ── Step 3a: OLS price prediction for next bar ─────────────────
        closes_arr = ohlc_df['close'].values[-20:].astype(float)
        x_arr = np.arange(len(closes_arr), dtype=float)
        _m, _b = np.polyfit(x_arr, closes_arr, 1)
        ols_predicted_price = float(_m * len(closes_arr) + _b)

        # ── Step 4: Risk ──────────────────────────────────────────────
        risk_assessment = self.risk_agent.analyze(
            ohlc_df, indicator_report, pattern_match, trend_report
        )

        # ── Step 5: Decision (algorithmic) ────────────────────────────
        decision = self.decision_agent.decide(
            ohlc_df, indicator_report, pattern_match, trend_report, risk_assessment
        )

        # ── Step 5b: Single combined LLM call (pattern + decision) ───
        llm_combined: dict = {}
        llm_error: str = ""
        if self.llm.available:
            try:
                chart_png = self._render_pattern_chart(ohlc_df)
                llm_combined = self.llm.analyze_and_decide(
                    chart_png_bytes=chart_png,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    entry=decision.entry_price,
                    algo_pattern=pattern_match.name,
                    algo_confidence=pattern_match.confidence,
                    indicator_overall=indicator_report.overall_signal,
                    rsi=indicator_report.rsi.value,
                    macd=indicator_report.macd.value,
                    roc=indicator_report.roc.value,
                    stoch_k=indicator_report.stoch.value,
                    willr=indicator_report.willr.value,
                    pattern_name=pattern_match.name,
                    pattern_dir=pattern_match.direction,
                    pattern_conf=pattern_match.confidence,
                    trend=trend_report.trend,
                    kappa=trend_report.kappa,
                    breakout_prob=trend_report.breakout_probability,
                    momentum_strength=trend_report.momentum_strength,
                    channel_width=trend_report.channel_width,
                    signals=trend_report.signals,
                    rr_ratio=risk_assessment.risk_reward_ratio,
                    sl=decision.stop_loss,
                    tp=decision.take_profit,
                    horizon_min=self.horizon_min,
                )
                if "error" in llm_combined:
                    llm_error = llm_combined.pop("error", "")
                elif not llm_combined.get("decision"):
                    # Call succeeded but every structured field came back empty —
                    # report it rather than letting the UI show a silent success.
                    llm_error = "LLM response contained no 'decision' field"
                    llm_combined = {}
                else:
                    direction = str(llm_combined["decision"]).upper()
                    if direction in ("LONG", "SHORT"):
                        decision.direction = direction
                    if "justification" in llm_combined:
                        decision.justification = llm_combined["justification"]
                    if "risk_reward_ratio" in llm_combined:
                        decision.risk_reward_ratio = float(llm_combined["risk_reward_ratio"])
                    # RiskAgent sized the bracket for the direction IT chose. If
                    # the LLM overturned that direction (or the R:R), the stop and
                    # target must be re-derived, or a SHORT setup renders with a
                    # LONG bracket — stop below entry, target above it.
                    if (decision.direction != risk_assessment.direction
                            or decision.risk_reward_ratio != risk_assessment.risk_reward_ratio):
                        self._reprice_brackets(decision, risk_assessment)
            except Exception as e:
                llm_combined = {}
                llm_error = str(e)

        # Split combined result for serialization compatibility
        llm_pattern = llm_combined  # has confirmed_pattern, structure, trend_context, symmetry
        llm_decision = llm_combined  # has decision, justification, watch_for, predicted_close_price

        elapsed = time.time() - t0

        return self._serialize(
            decision, indicator_report, pattern_match, trend_report,
            risk_assessment, ohlc_df, elapsed, llm_pattern, llm_decision,
            ols_predicted_price, "", llm_error
        )

    # ── Bracket repricing ─────────────────────────────────────────────────

    def _reprice_brackets(self, decision: TradeDecision, risk) -> None:
        """
        Recompute stop-loss and take-profit for the decision's final direction.

        Uses exactly RiskAgent's formula — stop at rho from entry, target at
        r x rho — so a decision the LLM leaves alone is unchanged. Both the
        decision and the risk assessment are updated together, so the two panes
        of the UI can never disagree about the same trade.
        """
        rho   = self.risk_agent.rho
        entry = decision.entry_price
        rr    = decision.risk_reward_ratio

        if decision.direction == "LONG":
            stop_loss   = entry * (1 - rho)
            take_profit = entry * (1 + rr * rho)
        else:
            stop_loss   = entry * (1 + rho)
            take_profit = entry * (1 - rr * rho)

        decision.stop_loss   = stop_loss
        decision.take_profit = take_profit

        risk.direction         = decision.direction
        risk.stop_loss         = stop_loss
        risk.take_profit       = take_profit
        risk.risk_reward_ratio = rr
        risk.risk_zone_width   = abs(entry - stop_loss)
        risk.reward_zone_width = abs(take_profit - entry)

        decision.trade_setup = self.decision_agent._build_trade_setup(
            decision.direction, entry, risk
        )

    # ── Serialization ─────────────────────────────────────────────────────

    def _serialize(self, decision: TradeDecision, ind, pat, tre, risk,
                   ohlc_df, elapsed, llm_pattern, llm_decision,
                   ols_predicted_price: float = 0.0,
                   llm_pattern_error: str = "",
                   llm_decision_error: str = "") -> Dict:

        closes = ohlc_df['close'].values.astype(float)
        highs  = ohlc_df['high'].values.astype(float)
        lows   = ohlc_df['low'].values.astype(float)
        opens  = ohlc_df['open'].values.astype(float)

        N = min(20, len(closes))
        x = np.arange(N, dtype=float)
        resist_y = (tre.resistance_line.slope * x + tre.resistance_line.intercept).tolist()
        support_y = (tre.support_line.slope * x + tre.support_line.intercept).tolist()

        # Timestamp column (may or may not exist)
        if 'timestamp' in ohlc_df.columns:
            timestamps = ohlc_df['timestamp'].values[-100:].tolist()
        else:
            timestamps = list(range(min(100, len(closes))))

        llm_predicted_price = llm_decision.get("predicted_close_price")
        try:
            llm_predicted_price = float(llm_predicted_price) if llm_predicted_price else None
        except (TypeError, ValueError):
            llm_predicted_price = None

        return {
            "timestamp":   time.time(),
            "timeframe":   self.timeframe,
            "symbol":      self.symbol,
            "elapsed_ms":  round(elapsed * 1000, 1),
            "llm_enabled": self.llm.available,
            "llm_errors": {
                "pattern": llm_pattern_error or None,
                "decision": llm_decision_error or None,
            },
            "predictions": {
                "current_price":      round(float(closes[-1]), 2),
                "ols_predicted_price": round(ols_predicted_price, 2),
                "llm_predicted_price": round(llm_predicted_price, 2) if llm_predicted_price else None,
            },

            "decision": {
                "direction":        decision.direction,
                "forecast_horizon": f"Next 3 × {self.timeframe} ({self.horizon_min} min)",
                "justification":    decision.justification,
                "trade_setup":      decision.trade_setup,
                "reflection":       decision.reflection,
                "risk_reward_ratio": round(decision.risk_reward_ratio, 3),
                "stop_loss":        round(decision.stop_loss, 2),
                "take_profit":      round(decision.take_profit, 2),
                "entry_price":      round(decision.entry_price, 2),
                "confidence":       round(decision.confidence, 3),
                "signal_alignment": decision.signal_alignment,
                "llm_watch_for":    llm_decision.get("watch_for", ""),
                # New richer LLM fields
                "llm_trade_type":        llm_pattern.get("trade_type", ""),
                "llm_entry_zone":        llm_pattern.get("entry_zone", ""),
                "llm_key_levels":        llm_pattern.get("key_levels", ""),
                "llm_signal_confluence": llm_pattern.get("signal_confluence"),
            },

            "indicators": {
                "rsi":   {"value": round(ind.rsi.value, 2),   "signal": ind.rsi.signal,   "description": ind.rsi.description,   "llm_analysis": llm_pattern.get("rsi_analysis", "")},
                "macd":  {"value": round(ind.macd.value, 6),  "signal": ind.macd.signal,  "description": ind.macd.description,  "llm_analysis": llm_pattern.get("macd_analysis", "")},
                "roc":   {"value": round(ind.roc.value, 4),   "signal": ind.roc.signal,   "description": ind.roc.description,   "llm_analysis": llm_pattern.get("roc_analysis", "")},
                "stoch": {"value": round(ind.stoch.value, 2), "signal": ind.stoch.signal, "description": ind.stoch.description, "llm_analysis": llm_pattern.get("stoch_analysis", "")},
                "willr": {"value": round(ind.willr.value, 2), "signal": ind.willr.signal, "description": ind.willr.description, "llm_analysis": llm_pattern.get("willr_analysis", "")},
                "overall":    ind.overall_signal,
                "confidence": round(ind.confidence, 3),
                "summary":    ind.summary,
                "llm_narrative": llm_pattern.get("indicator_narrative", ""),
            },

            "pattern": {
                "name":              pat.name,
                "direction":         pat.direction,
                "confidence":        round(pat.confidence, 3),
                "reliability":       pat.reliability,
                "structure_summary": pat.structure_summary,
                "trend_summary":     pat.trend_summary,
                "symmetry_summary":  pat.symmetry_summary,
                "pivot_points":      pat.pivot_points,
                # LLM enrichment
                "llm_confirmed":      llm_pattern.get("confirmed_pattern", ""),
                "llm_structure":      llm_pattern.get("structure", ""),
                "llm_trend":          llm_pattern.get("trend_context", ""),
                "llm_symmetry":       llm_pattern.get("symmetry", ""),
                "llm_target_price":   llm_pattern.get("pattern_target_price"),
            },

            "trend": {
                "classification":     tre.trend,
                "kappa":              round(tre.kappa, 8),
                "momentum_strength":  tre.momentum_strength,
                "prediction":         tre.prediction,
                "reasoning":          tre.reasoning,
                "signals":            tre.signals,
                "channel_width":      round(tre.channel_width, 2),
                "breakout_probability": round(tre.breakout_probability, 3),
                "llm_signal_label":   llm_pattern.get("trend_signal_label", ""),
                "llm_adx_strength":   llm_pattern.get("adx_strength", ""),
                "llm_momentum_note":  llm_pattern.get("momentum_note", ""),
                "llm_narrative":      llm_pattern.get("trend_narrative", ""),
                "resistance": {
                    "slope":     round(tre.resistance_line.slope, 8),
                    "intercept": round(tre.resistance_line.intercept, 2),
                    "direction": tre.resistance_line.direction,
                    "strength":  tre.resistance_line.strength,
                    "r_squared": round(tre.resistance_line.r_squared, 3),
                    "series":    resist_y,
                },
                "support": {
                    "slope":     round(tre.support_line.slope, 8),
                    "intercept": round(tre.support_line.intercept, 2),
                    "direction": tre.support_line.direction,
                    "strength":  tre.support_line.strength,
                    "r_squared": round(tre.support_line.r_squared, 3),
                    "series":    support_y,
                },
                "pivot_highs": tre.pivot_highs,
                "pivot_lows":  tre.pivot_lows,
            },

            "risk": {
                "direction":         risk.direction,
                "entry_price":       round(risk.entry_price, 2),
                "stop_loss":         round(risk.stop_loss, 2),
                "take_profit":       round(risk.take_profit, 2),
                "risk_reward_ratio": round(risk.risk_reward_ratio, 3),
                "risk_zone_width":   round(risk.risk_zone_width, 2),
                "reward_zone_width": round(risk.reward_zone_width, 2),
                "overall_risk_score": round(risk.overall_risk_score, 3),
                "radar": {
                    "momentum_strength":   round(risk.radar.momentum_strength, 3),
                    "volatility_risk":     round(risk.radar.volatility_risk, 3),
                    "sr_strength":         round(risk.radar.sr_strength, 3),
                    "breakout_likelihood": round(risk.radar.breakout_likelihood, 3),
                    "trend_certainty":     round(risk.radar.trend_certainty, 3),
                    "pattern_reliability": round(risk.radar.pattern_reliability, 3),
                },
            },

            "ohlc": {
                "open":       opens[-100:].tolist(),
                "high":       highs[-100:].tolist(),
                "low":        lows[-100:].tolist(),
                "close":      closes[-100:].tolist(),
                "timestamps": timestamps,
                "n":          min(100, len(closes)),
            },
        }

    # ── Chart rendering (for LLM pattern vision) ─────────────────────────

    def _render_pattern_chart(self, df: pd.DataFrame) -> bytes:
        """Render a minimal candlestick PNG for the pattern LLM prompt."""
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        window = df.tail(50)
        closes = window["close"].values
        highs  = window["high"].values
        lows   = window["low"].values
        opens  = window["open"].values
        n = len(closes)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")

        for i in range(n):
            color = "#26a641" if closes[i] >= opens[i] else "#e05252"
            ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8)
            body_lo = min(opens[i], closes[i])
            body_hi = max(opens[i], closes[i])
            rect = mpatches.FancyBboxPatch(
                (i - 0.3, body_lo), 0.6, max(body_hi - body_lo, 0.001),
                boxstyle="square,pad=0", linewidth=0, facecolor=color
            )
            ax.add_patch(rect)

        ax.set_xlim(-1, n)
        ax.tick_params(colors="#888", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_title(f"{self.symbol} — {self.timeframe}", color="#ccc", fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
