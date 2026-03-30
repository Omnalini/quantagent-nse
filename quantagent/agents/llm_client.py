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

COMBINED_PROMPT = """You are an expert HFT analyst for NSE Indian equities, simultaneously embodying four specialized QuantAgents. Reason as each agent independently, then synthesize into one trade decision.

Asset: {symbol} | Timeframe: {timeframe} | Current Entry: ₹{entry:.2f}
A {timeframe} candlestick chart is attached — study wicks, body sizes, and pattern geometry carefully.

=== PRE-COMPUTED AGENT INPUTS ===

[IndicatorAgent — momentum & oscillators]
RSI(14): {rsi:.2f} | MACD line: {macd:.4f} | ROC: {roc:.4f}
Stochastic %K: {stoch_k:.2f} | Williams %R: {willr:.2f}
Algo overall signal: {indicator_overall}

[PatternAgent — chart structure]
Detected: {algo_pattern} (confidence {confidence:.0%}) | Direction: {pattern_dir} | Reliability: {pattern_conf:.0%}

[TrendAgent — price dynamics via OLS]
Trend: {trend} | κ={kappa:.5f} | Breakout prob: {breakout:.0%}
Momentum strength: {momentum_strength} | Channel width: ₹{channel_width:.2f}
Active signals: {signals}

[RiskAgent — setup quality]
Algo R/R: {rr_ratio:.2f} | Stop-loss ₹{sl:.2f} | Take-profit ₹{tp:.2f}

=== INSTRUCTIONS ===
1. Study the candlestick chart image first — confirm or reject the algo pattern visually.
2. Choose LONG or SHORT (NEVER HOLD). Horizon: next 3×{timeframe} (~{horizon_min} min).
3. Predict the close price for the VERY NEXT single {timeframe} bar only.
4. Compute signal_confluence: count how many of the 5 indicators + pattern + trend align with your decision, divide by 7, multiply by 100 → integer.
5. Compute pattern_target_price using standard measurement rule (e.g. Double Top/Bottom: project neckline distance; Triangle: project at breakout; Head & Shoulders: project head height from neckline).
6. Identify the two most critical price levels (nearest support below, nearest resistance above) from the chart.

Reply in strict JSON only (no markdown fence, no prose outside braces):
{{
  "rsi_analysis": "<one precise sentence: value, zone, momentum implication>",
  "macd_analysis": "<one precise sentence: crossover state, histogram direction, momentum>",
  "roc_analysis": "<one precise sentence: rate of change magnitude and trend implication>",
  "stoch_analysis": "<one precise sentence: %K level, overbought/oversold, divergence if any>",
  "willr_analysis": "<one precise sentence: %R level, reversal potential>",
  "indicator_narrative": "<3 sentences: synthesize all 5 indicators, note confluences and contradictions, state net bias>",
  "confirmed_pattern": "<exact pattern name or 'None'>",
  "direction": "<Bullish|Bearish|Neutral>",
  "structure": "<one sentence: describe key highs/lows/wicks visible in chart, mention price levels>",
  "trend_context": "<one sentence: trend before and during the pattern, slope direction>",
  "symmetry": "<one sentence: left-right symmetry quality and what it implies for reliability>",
  "pattern_confidence": <0.0-1.0>,
  "pattern_target_price": <numeric float — measured price target from pattern projection>,
  "trend_signal_label": "<Likely Uptrend Signal|Likely Downtrend Signal|Sideways Consolidation>",
  "adx_strength": "<Strong|Moderate|Weak>",
  "momentum_note": "<one sentence: momentum direction, exhaustion or acceleration signals>",
  "trend_narrative": "<3 sentences: kappa direction, breakout probability, channel context, support/resistance levels>",
  "signal_confluence": <integer 0-100 — % of the 7 sub-signals aligned with your decision>,
  "trade_type": "<Trend-Continuation|Counter-Trend-Reversal|Breakout-Entry|Range-Fade>",
  "key_levels": "<Support: ₹X | Resistance: ₹Y>",
  "entry_zone": "<exact price or range for ideal entry, e.g. ₹1348–₹1352 on pullback>",
  "decision": "<LONG|SHORT>",
  "justification": "<3 sentences citing the 3 strongest cross-agent confluences that drove this decision>",
  "risk_reward_ratio": <float 1.2-2.5>,
  "watch_for": "<one specific price event that would invalidate this trade — include a price level>",
  "predicted_close_price": <numeric float — your best estimate of the next single {timeframe} bar close>
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
        result = client.analyze_and_decide(png_bytes, ...)
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

    def analyze_and_decide(
        self,
        chart_png_bytes: bytes,
        symbol: str, timeframe: str, entry: float,
        algo_pattern: str, algo_confidence: float,
        indicator_overall: str, rsi: float, macd: float,
        roc: float = 0.0, stoch_k: float = 50.0, willr: float = -50.0,
        pattern_name: str = "", pattern_dir: str = "Neutral", pattern_conf: float = 0.5,
        trend: str = "Sideways", kappa: float = 0.0, breakout_prob: float = 0.0,
        momentum_strength: str = "Moderate", channel_width: float = 0.0, signals: str = "",
        rr_ratio: float = 1.5, sl: float = 0.0, tp: float = 0.0,
        horizon_min: int = 45,
    ) -> dict:
        """Single combined call: all four agent narratives + final trade decision."""
        prompt = COMBINED_PROMPT.format(
            symbol=symbol, timeframe=timeframe, entry=entry,
            algo_pattern=algo_pattern, confidence=algo_confidence,
            indicator_overall=indicator_overall, rsi=rsi, macd=macd,
            roc=roc, stoch_k=stoch_k, willr=willr,
            pattern_name=pattern_name, pattern_dir=pattern_dir,
            pattern_conf=pattern_conf, trend=trend, kappa=kappa,
            breakout=breakout_prob, momentum_strength=momentum_strength,
            channel_width=channel_width, signals=signals,
            rr_ratio=rr_ratio, sl=sl, tp=tp,
            horizon_min=horizon_min,
        )
        if self._ant:
            return self._ant_vision(chart_png_bytes, prompt)
        if self._gem:
            return self._gem_vision(chart_png_bytes, prompt)
        return {}

    def _safe_gem_text(self, resp) -> str:
        """
        Extract output text from a Gemini response.
        For thinking models (gemini-2.5-flash), resp.text may return the
        internal reasoning (no JSON) instead of the actual output.
        We iterate parts directly and skip thought parts to get the real output.
        """
        # Iterate parts first — explicitly skip thought parts
        try:
            parts = resp.candidates[0].content.parts
            for part in reversed(parts):
                text = getattr(part, 'text', None)
                is_thought = getattr(part, 'thought', False)
                if text and not is_thought:
                    return text.strip()
        except Exception:
            pass
        # Fallback: SDK resp.text (may include thinking content on some SDK versions)
        try:
            if resp.text:
                return resp.text.strip()
        except Exception:
            pass
        return ""

    def _extract_json(self, raw: str) -> dict:
        """
        Robustly extract a JSON object from model response.
        Handles: markdown fences, thinking prose before/after, stray text.
        """
        if not raw:
            return {"error": "Empty response from model"}
        # Find outermost { ... }
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError as e:
                return {"error": f"JSON parse failed: {e}", "preview": raw[start:start+300]}
        return {"error": "No JSON object in response", "preview": raw[:300]}

    def ping(self) -> dict:
        """Quick connectivity check. Returns {"ok": bool, "provider": str, "reply"/"error": str}."""
        if self._gem:
            try:
                resp = self._gem.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Reply with the single word: ok",
                    config=_gtypes.GenerateContentConfig(max_output_tokens=64, temperature=0),
                )
                text = self._safe_gem_text(resp)
                return {"ok": True, "provider": "gemini", "reply": text or "(empty response)"}
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
                max_tokens=2048,
                messages=[{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            return self._extract_json(resp.content[0].text.strip())
        except Exception as e:
            return {"error": str(e)}

    def _ant_text(self, prompt: str) -> dict:
        try:
            resp = self._ant.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._extract_json(resp.content[0].text.strip())
        except Exception as e:
            return {"error": str(e)}

    # ── Gemini backend ────────────────────────────────────────────────────

    def _gem_vision(self, png: bytes, prompt: str) -> dict:
        try:
            img_part = _gtypes.Part.from_bytes(data=png, mime_type="image/png")
            cfg = {"max_output_tokens": 2048, "temperature": 0.2}
            try:
                cfg["thinking_config"] = _gtypes.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass  # older SDK without ThinkingConfig
            resp = self._gem.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img_part, prompt],
                config=_gtypes.GenerateContentConfig(**cfg),
            )
            raw = self._safe_gem_text(resp)
            return self._extract_json(raw)
        except Exception as e:
            return {"error": str(e)}

    def _gem_text(self, prompt: str) -> dict:
        try:
            cfg = {"max_output_tokens": 768, "temperature": 0.2}
            try:
                cfg["thinking_config"] = _gtypes.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
            resp = self._gem.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=_gtypes.GenerateContentConfig(**cfg),
            )
            raw = self._safe_gem_text(resp)
            return self._extract_json(raw)
        except Exception as e:
            return {"error": str(e)}
