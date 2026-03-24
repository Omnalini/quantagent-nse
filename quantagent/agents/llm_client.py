"""
LLM Client for QuantAgent
Wraps the Anthropic Claude API (claude-haiku-4-5 by default).
Falls back gracefully to None when no API key is provided.

Token-saving strategies implemented here:
  - Use claude-haiku-4-5-20251001 (cheapest model, fast)
  - Short, focused prompts
  - max_tokens capped at 512 for pattern + 768 for decision
  - Skip LLM entirely when api_key is None
"""

from __future__ import annotations
import base64
import io
import json
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# Default model — cheapest Claude model (haiku)
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ── Prompt templates ─────────────────────────────────────────────────────

PATTERN_PROMPT = """You are a chart-pattern expert for Indian equity markets (NSE/Nifty 50).
A {timeframe} candlestick chart of {symbol} is attached.

The algorithmic pre-scan detected: **{algo_pattern}** (confidence {confidence:.0%}).

Examine the chart and answer concisely in JSON only (no markdown):
{{
  "confirmed_pattern": "<pattern name or 'None'>",
  "direction": "<Bullish|Bearish|Neutral>",
  "structure": "<one sentence on highs/lows/shape>",
  "trend_context": "<one sentence on surrounding trend>",
  "symmetry": "<one sentence on shape symmetry>",
  "confidence": <0.0-1.0>
}}"""

DECISION_PROMPT = """You are a high-frequency trading analyst for NSE Indian equities.
Asset: {symbol} | Timeframe: {timeframe} | Entry price: ₹{entry:.2f}

Upstream agent summary:
- Indicators  : {indicator_overall} (RSI {rsi:.1f}, MACD {macd:.4f})
- Pattern     : {pattern_name} ({pattern_dir}, conf {pattern_conf:.0%})
- Trend       : {trend} (κ={kappa:.5f}, breakout_prob {breakout:.0%})
- Risk/Reward : {rr_ratio:.2f} | Stop ₹{sl:.2f} | Target ₹{tp:.2f}

You MUST choose LONG or SHORT (no HOLD).
Forecast horizon: next 3 × {timeframe} bars (~{horizon_min} minutes).

Reply in JSON only (no markdown):
{{
  "decision": "<LONG|SHORT>",
  "justification": "<2-3 sentences citing the strongest signals>",
  "risk_reward_ratio": <1.2-1.8>,
  "watch_for": "<one invalidation signal to monitor>"
}}"""


class LLMClient:
    """
    Thin wrapper around the Anthropic Messages API.

    Usage:
        client = LLMClient(api_key="sk-ant-...")
        pattern_json = client.analyze_pattern(chart_png_bytes, ...)
        decision_json = client.synthesize_decision(...)
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self._client: Optional["anthropic.Anthropic"] = None

        if api_key and _ANTHROPIC_OK:
            self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── Pattern analysis (multimodal) ────────────────────────────────────

    def analyze_pattern(
        self,
        chart_png_bytes: bytes,
        symbol: str,
        timeframe: str,
        algo_pattern: str,
        algo_confidence: float,
    ) -> dict:
        """
        Send a chart image to Claude and get pattern recognition JSON back.
        Returns the parsed dict, or {} if LLM unavailable.
        """
        if not self.available:
            return {}

        b64 = base64.standard_b64encode(chart_png_bytes).decode("utf-8")
        prompt = PATTERN_PROMPT.format(
            timeframe=timeframe,
            symbol=symbol,
            algo_pattern=algo_pattern,
            confidence=algo_confidence,
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            raw = response.content[0].text.strip()
            return json.loads(raw)
        except Exception as exc:
            return {"error": str(exc)}

    # ── Decision synthesis (text only) ───────────────────────────────────

    def synthesize_decision(
        self,
        symbol: str,
        timeframe: str,
        entry: float,
        indicator_overall: str,
        rsi: float,
        macd: float,
        pattern_name: str,
        pattern_dir: str,
        pattern_conf: float,
        trend: str,
        kappa: float,
        breakout_prob: float,
        rr_ratio: float,
        sl: float,
        tp: float,
        horizon_min: int = 45,
    ) -> dict:
        """
        Ask Claude to synthesize a final LONG/SHORT decision.
        Returns parsed dict, or {} if LLM unavailable.
        """
        if not self.available:
            return {}

        prompt = DECISION_PROMPT.format(
            symbol=symbol,
            timeframe=timeframe,
            entry=entry,
            indicator_overall=indicator_overall,
            rsi=rsi,
            macd=macd,
            pattern_name=pattern_name,
            pattern_dir=pattern_dir,
            pattern_conf=pattern_conf,
            trend=trend,
            kappa=kappa,
            breakout=breakout_prob,
            rr_ratio=rr_ratio,
            sl=sl,
            tp=tp,
            horizon_min=horizon_min,
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            return json.loads(raw)
        except Exception as exc:
            return {"error": str(exc)}
