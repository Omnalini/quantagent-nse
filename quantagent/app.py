"""
QuantAgent Flask Backend — Nifty 50 Edition
Real market data via yfinance + optional Claude LLM enhancement.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import uuid
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request, render_template, Response, session
from flask_cors import CORS

from agents import QuantAgent
from data.nifty_fetcher import (
    fetch_ohlc_cached, get_nifty_50_list, is_market_open, NIFTY_50
)
from data.simulator import MarketSimulator, AccuracyTracker
from charts import generate_all_charts

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# ── Global session state ──────────────────────────────────────────────────
trackers: dict = {}
session_meta: dict = {}

DEFAULT_SYMBOL    = "RELIANCE"
DEFAULT_INTERVAL  = "15m"
DEFAULT_PERIOD    = "5d"


# ── Helpers ───────────────────────────────────────────────────────────────

def _get_api_key(request_data: dict) -> str | None:
    """Read API key from request body (never from URL params)."""
    return request_data.get("api_key") or request.headers.get("X-Api-Key")


def _fetch_df(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch real NSE data; fall back to simulator on any error."""
    try:
        df = fetch_ohlc_cached(symbol, interval, period)
        if len(df) >= 30:
            return df
    except Exception:
        pass
    # Fallback: synthetic data if yfinance fails
    sim = MarketSimulator(asset="NQ")
    sim.generate_history(200)
    return sim.get_dataframe()


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stocks', methods=['GET'])
def list_stocks():
    """Return all Nifty 50 stocks."""
    return jsonify({"stocks": get_nifty_50_list()})


@app.route('/api/market_status', methods=['GET'])
def market_status():
    """Return whether NSE is currently open."""
    return jsonify({
        "is_open": is_market_open(),
        "exchange": "NSE",
        "hours": "09:15 – 15:30 IST (Mon–Fri)"
    })


@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    """
    Store the Anthropic API key in the server-side session.
    Key is never written to disk or logged.
    """
    data = request.json or {}
    key = data.get("api_key", "").strip()
    if not key:
        return jsonify({"error": "api_key is required"}), 400
    if key.startswith("sk-ant-"):
        provider = "anthropic"
    elif key.startswith("AIza"):
        provider = "gemini"
    else:
        return jsonify({"error": "Unrecognised key format. Use sk-ant-... (Claude) or AIza... (Gemini)"}), 400
    session["api_key"] = key
    return jsonify({"status": "ok", "llm_enabled": True, "provider": provider})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Run full QuantAgent pipeline on a Nifty 50 stock.

    POST body:
      symbol    : NSE ticker, e.g. "RELIANCE"
      interval  : "15m" (default)
      period    : "5d"  (default)
      api_key   : Anthropic key (optional — enables LLM enhancement)
      session_id: tracking ID (optional)
    """
    data = request.json or {}
    symbol     = data.get("symbol", DEFAULT_SYMBOL).upper()
    interval   = data.get("interval", DEFAULT_INTERVAL)
    period     = data.get("period", DEFAULT_PERIOD)
    session_id = data.get("session_id", str(uuid.uuid4()))
    api_key    = _get_api_key(data) or session.get("api_key")

    df = _fetch_df(symbol, interval, period)
    if len(df) < 30:
        return jsonify({"error": "Not enough data bars (need ≥30)"}), 400

    # Compute horizon minutes: 3 candles × bar_minutes
    bar_min_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    bar_min = bar_min_map.get(interval, 15)
    horizon_min = bar_min * 3

    agent = QuantAgent(timeframe=interval, api_key=api_key,
                       symbol=symbol, horizon_min=horizon_min)
    result = agent.run(df)

    if "error" in result:
        return jsonify(result), 400

    # Accuracy tracking
    if session_id not in trackers:
        trackers[session_id] = AccuracyTracker()
    tracker = trackers[session_id]
    pred_id = str(uuid.uuid4())[:8]
    tracker.record_prediction(
        prediction_id=pred_id,
        direction=result['decision']['direction'],
        entry_price=result['decision']['entry_price'],
        stop_loss=result['decision']['stop_loss'],
        take_profit=result['decision']['take_profit'],
        timestamp=time.time()
    )

    session_meta[session_id] = {
        "symbol": symbol, "interval": interval,
        "prediction_count": session_meta.get(session_id, {}).get("prediction_count", 0) + 1
    }

    result["prediction_id"] = pred_id
    result["session_id"]    = session_id
    result["symbol"]        = symbol
    result["llm_used"]      = bool(api_key)
    result["market_open"]   = is_market_open()

    return jsonify(result)


@app.route('/api/quote', methods=['GET'])
def quote():
    """Return the latest OHLCV bars for a symbol (for price widget)."""
    symbol   = request.args.get("symbol", DEFAULT_SYMBOL).upper()
    interval = request.args.get("interval", DEFAULT_INTERVAL)
    period   = request.args.get("period", "1d")
    n        = int(request.args.get("n", 50))

    try:
        df = fetch_ohlc_cached(symbol, interval, period)
        last = df.tail(n)
        return jsonify({
            "symbol": symbol,
            "bars": last[["open","high","low","close","volume","timestamp"]].to_dict("records"),
            "latest_price": float(df["close"].iloc[-1]),
            "bar_count": len(df),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    session_id = request.args.get("session_id", "default")
    if session_id not in trackers:
        return jsonify({"total": 0, "validated": 0, "accuracy_pct": 0.0}), 200
    return jsonify(trackers[session_id].get_summary())


@app.route('/api/stream', methods=['GET'])
def stream():
    """
    SSE stream: push price ticks every 30 s from yfinance (1m bars).
    LLM analysis is NOT triggered here to conserve API credits.
    Auto-analysis (rule-based only) fires every 10 ticks.
    """
    symbol     = request.args.get("symbol", DEFAULT_SYMBOL).upper()
    interval   = request.args.get("interval", DEFAULT_INTERVAL)
    session_id = request.args.get("session_id", "default")

    def generate():
        agent = QuantAgent(timeframe=interval)
        tick = 0
        while True:
            try:
                df = fetch_ohlc_cached(symbol, "1m", "1d")
                latest = df.iloc[-1]
                event = {
                    "type": "tick",
                    "symbol": symbol,
                    "price": float(latest["close"]),
                    "open": float(latest["open"]),
                    "high": float(latest["high"]),
                    "low": float(latest["low"]),
                    "volume": float(latest.get("volume", 0)),
                    "timestamp": float(latest.get("timestamp", time.time())),
                    "market_open": is_market_open(),
                }

                tick += 1
                if tick % 10 == 0:
                    full_df = fetch_ohlc_cached(symbol, interval, "5d")
                    if len(full_df) >= 30:
                        analysis = agent.run(full_df)
                        if "decision" in analysis:
                            event["type"]     = "analysis"
                            event["analysis"] = analysis

                yield f"data: {json.dumps(event)}\n\n"
                time.sleep(30)
            except GeneratorExit:
                break
            except Exception as exc:
                yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"
                time.sleep(30)

    return Response(generate(), mimetype='text/event-stream',
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    Quick backtest on cached historical 15m data (rule-based, no LLM).
    POST: { symbol, interval, period, analysis_every }
    """
    data          = request.json or {}
    symbol        = data.get("symbol", DEFAULT_SYMBOL).upper()
    interval      = data.get("interval", DEFAULT_INTERVAL)
    period        = data.get("period", "1mo")
    analysis_every = int(data.get("analysis_every", 10))

    try:
        df = fetch_ohlc_cached(symbol, interval, period)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if len(df) < 50:
        return jsonify({"error": "Not enough bars for backtest"}), 400

    tracker = AccuracyTracker()
    agent   = QuantAgent(timeframe=interval)
    results = []

    for i in range(50, len(df), analysis_every):
        window = df.iloc[:i]
        analysis = agent.run(window)
        if "decision" not in analysis:
            continue

        pred_id = f"bt_{i}"
        future_bar = df.iloc[i] if i < len(df) else None
        tracker.record_prediction(
            prediction_id=pred_id,
            direction=analysis["decision"]["direction"],
            entry_price=analysis["decision"]["entry_price"],
            stop_loss=analysis["decision"]["stop_loss"],
            take_profit=analysis["decision"]["take_profit"],
            timestamp=time.time()
        )
        if future_bar is not None:
            tracker.add_validation_bar(pred_id, future_bar.to_dict())

        results.append({
            "bar_idx":   i,
            "datetime":  str(df.iloc[i - 1].get("datetime", "")),
            "direction": analysis["decision"]["direction"],
            "confidence": analysis["decision"]["confidence"],
            "pattern":   analysis["pattern"]["name"],
            "trend":     analysis["trend"]["classification"],
            "price":     analysis["decision"]["entry_price"],
        })

    return jsonify({
        "symbol":             symbol,
        "interval":           interval,
        "backtest_summary":   tracker.get_summary(),
        "predictions_made":   results,
        "total_bars":         len(df),
    })


if __name__ == '__main__':
    print("QuantAgent Nifty 50 — starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=3000, threaded=True)
