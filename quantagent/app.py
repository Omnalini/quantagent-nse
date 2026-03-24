"""
QuantAgent Flask Backend
Provides REST API endpoints for all agent functionality + real-time simulation
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
from flask import Flask, jsonify, request, render_template, Response
from agents import QuantAgent
from data.simulator import MarketSimulator, AccuracyTracker
from charts import generate_all_charts

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global state
simulators: dict = {}
trackers: dict = {}
session_data: dict = {}
prediction_history: dict = {}

DEFAULT_ASSET = "SPX"
DEFAULT_TF = "15m"


def get_or_create_session(session_id: str, asset: str = DEFAULT_ASSET):
    """Get or create a simulation session"""
    if session_id not in simulators:
        sim = MarketSimulator(asset=asset)
        sim.generate_history(200)  # Initialize with 200 bars
        simulators[session_id] = sim
        trackers[session_id] = AccuracyTracker()
        session_data[session_id] = {
            "asset": asset,
            "timeframe": DEFAULT_TF,
            "prediction_count": 0,
            "last_prediction_id": None
        }
    return simulators[session_id], trackers[session_id]


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/assets', methods=['GET'])
def get_assets():
    """Return available assets"""
    return jsonify({"assets": list(MarketSimulator.ASSETS.keys())})


@app.route('/api/init', methods=['POST'])
def init_session():
    """Initialize or reset a trading session"""
    data = request.json or {}
    session_id = data.get('session_id', str(uuid.uuid4()))
    asset = data.get('asset', DEFAULT_ASSET)
    timeframe = data.get('timeframe', DEFAULT_TF)
    seed = data.get('seed', None)

    # Create fresh simulator
    sim = MarketSimulator(asset=asset, seed=seed)
    sim.generate_history(200)
    simulators[session_id] = sim
    trackers[session_id] = AccuracyTracker()
    session_data[session_id] = {
        "asset": asset,
        "timeframe": timeframe,
        "prediction_count": 0,
        "last_prediction_id": None
    }

    return jsonify({
        "session_id": session_id,
        "asset": asset,
        "timeframe": timeframe,
        "bars_initialized": 200,
        "current_price": sim.get_current_price()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Run full QuantAgent analysis on current data.
    POST body: { session_id, asset, timeframe }
    """
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    asset = data.get('asset', DEFAULT_ASSET)
    timeframe = data.get('timeframe', DEFAULT_TF)

    sim, tracker = get_or_create_session(session_id, asset)
    df = sim.get_dataframe()

    if df.empty or len(df) < 30:
        return jsonify({"error": "Not enough data"}), 400

    # Run QuantAgent pipeline
    agent = QuantAgent(timeframe=timeframe)
    result = agent.run(df)

    if "error" in result:
        return jsonify(result), 400

    # Record prediction for accuracy tracking
    pred_id = str(uuid.uuid4())[:8]
    tracker.record_prediction(
        prediction_id=pred_id,
        direction=result['decision']['direction'],
        entry_price=result['decision']['entry_price'],
        stop_loss=result['decision']['stop_loss'],
        take_profit=result['decision']['take_profit'],
        timestamp=time.time()
    )

    session_data[session_id]['prediction_count'] += 1
    session_data[session_id]['last_prediction_id'] = pred_id
    result['prediction_id'] = pred_id
    result['session_info'] = session_data[session_id]

    return jsonify(result)


@app.route('/api/tick', methods=['POST'])
def get_tick():
    """
    Advance simulation by one bar and validate pending predictions.
    POST body: { session_id }
    Returns: new bar + accuracy update
    """
    data = request.json or {}
    session_id = data.get('session_id', 'default')

    if session_id not in simulators:
        return jsonify({"error": "Session not found"}), 404

    sim = simulators[session_id]
    tracker = trackers[session_id]

    # Generate next bar
    new_bar = sim.get_next_bar()

    # Add to all pending predictions for validation
    pending_pred_id = session_data[session_id].get('last_prediction_id')
    if pending_pred_id:
        # Add validation bar to all unvalidated predictions
        for pred in tracker.predictions:
            if not pred.get('validated', True):
                tracker.add_validation_bar(pred['id'], new_bar)

    # Current price info
    df = sim.get_dataframe()
    last_bars = df.tail(5).to_dict('records')

    accuracy_summary = tracker.get_summary()

    return jsonify({
        "new_bar": new_bar,
        "current_price": sim.get_current_price(),
        "total_bars": len(sim.history),
        "accuracy": accuracy_summary,
        "last_bars": last_bars
    })


@app.route('/api/stream')
def stream():
    """
    Server-Sent Events stream for real-time simulation.
    Pushes new bars every 2 seconds + auto-analysis every 10 bars.
    """
    session_id = request.args.get('session_id', 'default')
    asset = request.args.get('asset', DEFAULT_ASSET)
    timeframe = request.args.get('timeframe', DEFAULT_TF)

    def generate():
        sim, tracker = get_or_create_session(session_id, asset)
        bar_count = 0
        agent = QuantAgent(timeframe=timeframe)

        while True:
            try:
                # Generate new bar
                new_bar = sim.get_next_bar()
                bar_count += 1

                # Validate pending predictions
                for pred in tracker.predictions:
                    if not pred.get('validated', False):
                        tracker.add_validation_bar(pred['id'], new_bar)

                event_data = {
                    "type": "bar",
                    "bar": new_bar,
                    "current_price": sim.get_current_price(),
                    "bar_count": bar_count,
                    "accuracy": tracker.get_summary()
                }

                # Auto-analyze every 10 bars
                if bar_count % 10 == 0:
                    df = sim.get_dataframe()
                    if len(df) >= 30:
                        analysis = agent.run(df)
                        if "decision" in analysis:
                            pred_id = str(uuid.uuid4())[:8]
                            tracker.record_prediction(
                                prediction_id=pred_id,
                                direction=analysis['decision']['direction'],
                                entry_price=analysis['decision']['entry_price'],
                                stop_loss=analysis['decision']['stop_loss'],
                                take_profit=analysis['decision']['take_profit'],
                                timestamp=time.time()
                            )
                            session_data[session_id]['last_prediction_id'] = pred_id
                            analysis['prediction_id'] = pred_id
                            event_data["analysis"] = analysis
                            event_data["type"] = "analysis"

                yield f"data: {json.dumps(event_data)}\n\n"
                time.sleep(2)

            except GeneratorExit:
                break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                time.sleep(2)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    """Get accuracy statistics for a session"""
    session_id = request.args.get('session_id', 'default')
    if session_id not in trackers:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(trackers[session_id].get_summary())


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get OHLC history for a session"""
    session_id = request.args.get('session_id', 'default')
    n = int(request.args.get('n', 100))

    if session_id not in simulators:
        return jsonify({"error": "Session not found"}), 404

    df = simulators[session_id].get_dataframe()
    last_n = df.tail(n)

    return jsonify({
        "bars": last_n.to_dict('records'),
        "total": len(df),
        "asset": session_data.get(session_id, {}).get('asset', DEFAULT_ASSET)
    })


@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    Run backtest: generate N bars, run analysis every M bars, report accuracy.
    POST body: { asset, n_bars, analysis_every, timeframe }
    """
    data = request.json or {}
    asset = data.get('asset', 'SPX')
    n_bars = min(int(data.get('n_bars', 500)), 2000)
    analysis_every = int(data.get('analysis_every', 10))
    timeframe = data.get('timeframe', '15m')
    seed = data.get('seed', 42)

    sim = MarketSimulator(asset=asset, seed=seed)
    sim.generate_history(100)  # Warm-up

    tracker = AccuracyTracker()
    agent = QuantAgent(timeframe=timeframe)
    results = []

    for i in range(n_bars):
        new_bar = sim.get_next_bar()

        # Add to pending predictions
        for pred in tracker.predictions:
            if not pred.get('validated', False):
                tracker.add_validation_bar(pred['id'], new_bar)

        if i % analysis_every == 0 and len(sim.history) >= 50:
            df = sim.get_dataframe()
            analysis = agent.run(df)

            if "decision" in analysis:
                pred_id = f"bt_{i}"
                tracker.record_prediction(
                    prediction_id=pred_id,
                    direction=analysis['decision']['direction'],
                    entry_price=analysis['decision']['entry_price'],
                    stop_loss=analysis['decision']['stop_loss'],
                    take_profit=analysis['decision']['take_profit'],
                    timestamp=time.time()
                )
                results.append({
                    "bar_idx": i,
                    "direction": analysis['decision']['direction'],
                    "confidence": analysis['decision']['confidence'],
                    "pattern": analysis['pattern']['name'],
                    "trend": analysis['trend']['classification'],
                    "indicator": analysis['indicators']['overall'],
                    "price": analysis['decision']['entry_price'],
                    "regime": new_bar.get('regime', 'Unknown')
                })

    summary = tracker.get_summary()
    return jsonify({
        "backtest_summary": summary,
        "predictions_made": results,
        "asset": asset,
        "n_bars": n_bars,
        "analysis_every": analysis_every
    })


@app.route('/api/charts', methods=['POST'])
def get_charts():
    """
    Generate matplotlib charts for all four agents.
    POST body: { session_id, asset, timeframe }
    Returns base64 PNG images for each agent.
    """
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    asset = data.get('asset', DEFAULT_ASSET)
    timeframe = data.get('timeframe', DEFAULT_TF)

    sim, _ = get_or_create_session(session_id, asset)
    df = sim.get_dataframe()
    if df.empty or len(df) < 30:
        return jsonify({"error": "Not enough data"}), 400

    agent = QuantAgent(timeframe=timeframe)
    result = agent.run(df)
    if "error" in result:
        return jsonify(result), 400

    charts = generate_all_charts(df, result)
    return jsonify({
        "charts": {k: v for k, v in charts.items() if v is not None},
        "analysis": result
    })


if __name__ == '__main__':
    print("🚀 QuantAgent Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
