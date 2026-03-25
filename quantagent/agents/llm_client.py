"""
LLM Client for QuantAgent — supports Anthropic Claude and Google Gemini.

Provider detection:
  - key starts with "sk-ant-"  → Anthropic Claude (claude-haiku-4-5-20251001)
  - key starts with "AIza"     → Google Gemini   (gemini-2.0-flash, free tier)

Token-saving strategies:
  - Cheapest model per provider (haiku / flash)
  - max_tokens capped at 512 (pattern) and 768 (decision)
  - LLM skipped entirely when no key is set
"""

from __future__ import annotations
import base64
import json
from typing import Optional

try:
    import anthropic as _ant
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    from google import genai as _genai
    from google.genai import types as _gtypes
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

# ── Prompt templates (shared between providers) ───────────────────────────

PATTERN_PROMPT = """You are a chart-pattern expert for Indian equity markets (NSE/Nifty 50).
A {timeframe} candlestick chart of {symbol} is attached.

Algorithmic pre-scan detected: {algo_pattern} (confidence {confidence:.0%}).

Examine the chart and reply in JSON only (no markdown fence):
{{
  "confirmed_pattern": "<pattern name or None>",
  "direction": "<Bullish|Bearish|Neutral>",
  "structure": "<one sentence on highs/lows/shape>",
  "trend_context": "<one sentence on surrounding trend>",
  "symmetry": "<one sentence on shape symmetry>",
  "confidence": <0.0-1.0>
}}"""

DECISION_PROMPT = """You are an HFT analyst for NSE Indian equities.
Asset: {symbol} | Timeframe: {timeframe} | Entry: ₹{entry:.2f}

Agent signals:
- Indicators : {indicator_overall} (RSI {rsi:.1f}, MACD {macd:.4f})
- Pattern    : {pattern_name} ({pattern_dir}, conf {pattern_conf:.0%})
- Trend      : {trend} (κ={kappa:.5f}, breakout_prob {breakout:.0%})
- Risk/Reward: {rr_ratio:.2f} | SL ₹{sl:.2f} | TP ₹{tp:.2f}

Choose LONG or SHORT (no HOLD). Horizon: next 3×{timeframe} (~{horizon_min} min).

Reply in JSON only (no markdown fence):
{{
  "decision": "<LONG|SHORT>",
  "justification": "<2-3 sentences citing strongest signals>",
  "risk_reward_ratio": <1.2-1.8>,
  "watch_for": "<one invalidation signal>"
}}"""


def _detect_provider(api_key: str) -> str:
    """Return 'anthropic', 'gemini', or 'unknown'."""
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("AIza"):
        return "gemini"
    return "unknown"


class LLMClient:
    """
    Unified LLM client. Automatically selects the backend from the key prefix.

    Usage:
        client = LLMClient(api_key="AIza...")        # Gemini
        client = LLMClient(api_key="sk-ant-...")     # Claude
        result = client.analyze_pattern(png_bytes, ...)
        result = client.synthesize_decision(...)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key
        self.provider = "none"
        self._ant     = None   # anthropic.Anthropic instance
        self._gem     = None   # google.genai.Client instance

        if not api_key:
            return

        self.provider = _detect_provider(api_key)

        if self.provider == "anthropic" and _ANTHROPIC_OK:
            self._ant = _ant.Anthropic(api_key=api_key)

        elif self.provider == "gemini" and _GEMINI_OK:
            self._gem = _genai.Client(api_key=api_key)

    @property
    def available(self) -> bool:
        return self._ant is not None or self._gem is not None

    # ── Public API ────────────────────────────────────────────────────────

    def analyze_pattern(
        self,
        chart_png_bytes: bytes,
        symbol: str,
        timeframe: str,
        algo_pattern: str,
        algo_confidence: float,
    ) -> dict:
        prompt = PATTERN_PROMPT.format(
            timeframe=timeframe, symbol=symbol,
            algo_pattern=algo_pattern, confidence=algo_confidence,
        )
        if self._ant:
            return self._ant_vision(chart_png_bytes, prompt)
        if self._gem:
            return self._gem_vision(chart_png_bytes, prompt)
        return {}

    def synthesize_decision(
        self,
        symbol: str, timeframe: str, entry: float,
        indicator_overall: str, rsi: float, macd: float,
        pattern_name: str, pattern_dir: str, pattern_conf: float,
        trend: str, kappa: float, breakout_prob: float,
        rr_ratio: float, sl: float, tp: float,
        horizon_min: int = 45,
    ) -> dict:
        prompt = DECISION_PROMPT.format(
            symbol=symbol, timeframe=timeframe, entry=entry,
            indicator_overall=indicator_overall, rsi=rsi, macd=macd,
            pattern_name=pattern_name, pattern_dir=pattern_dir,
            pattern_conf=pattern_conf, trend=trend, kappa=kappa,
            breakout=breakout_prob, rr_ratio=rr_ratio, sl=sl, tp=tp,
            horizon_min=horizon_min,
        )
        if self._ant:
            return self._ant_text(prompt)
        if self._gem:
            return self._gem_text(prompt)
        return {}

    def ping(self) -> dict:
        """Quick connectivity check. Returns {"ok": bool, "provider": str, "reply"/"error": str}."""
        if self._gem:
            try:
                resp = self._gem.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Reply with the single word: ok",
                    config=_gtypes.GenerateContentConfig(max_output_tokens=10, temperature=0),
                )
                return {"ok": True, "provider": "gemini", "reply": resp.text.strip()}
            except Exception as e:
                return {"ok": False, "provider": "gemini", "error": str(e)}
        if self._ant:
            try:
                resp = self._ant.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                )
                return {"ok": True, "provider": "anthropic", "reply": resp.content[0].text.strip()}
            except Exception as e:
                return {"ok": False, "provider": "anthropic", "error": str(e)}
        return {"ok": False, "provider": "none", "error": "No API client initialized"}

    # ── Anthropic backend ─────────────────────────────────────────────────

    def _ant_vision(self, png: bytes, prompt: str) -> dict:
        b64 = base64.standard_b64encode(png).decode()
        try:
            resp = self._ant.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            return json.loads(resp.content[0].text.strip())
        except Exception as e:
            return {"error": str(e)}

    def _ant_text(self, prompt: str) -> dict:
        try:
            resp = self._ant.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=768,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(resp.content[0].text.strip())
        except Exception as e:
            return {"error": str(e)}

    # ── Gemini backend ────────────────────────────────────────────────────

    def _gem_vision(self, png: bytes, prompt: str) -> dict:
        try:
            img_part = _gtypes.Part.from_bytes(data=png, mime_type="image/png")
            resp = self._gem.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img_part, prompt],
                config=_gtypes.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.2,
                ),
            )
            raw = resp.text.strip()
            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            return {"error": str(e)}

    def _gem_text(self, prompt: str) -> dict:
        try:
            resp = self._gem.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=_gtypes.GenerateContentConfig(
                    max_output_tokens=768,
                    temperature=0.2,
                ),
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            return {"error": str(e)}
