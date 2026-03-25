"""
Market Data Simulator
Generates realistic OHLC data with configurable market regimes.
Used when live data feeds are unavailable.
Also tracks prediction accuracy vs. actual price movement.
"""

import numpy as np
import pandas as pd
import time
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable


class MarketSimulator:
    """
    Generates realistic OHLC data simulating various market conditions:
    - Trending (bull/bear)
    - Ranging / sideways
    - Volatile / breakout
    Uses regime-switching model with occasional pattern formations.
    """

    ASSETS = {
        "BTC/USD": {"price": 65000, "vol": 0.025, "tick": 10.0},
        "NQ":      {"price": 21000, "vol": 0.012, "tick": 0.25},
        "SPX":     {"price": 5200,  "vol": 0.008, "tick": 0.25},
        "ES":      {"price": 5195,  "vol": 0.009, "tick": 0.25},
        "QQQ":     {"price": 450,   "vol": 0.011, "tick": 0.01},
        "AAPL":    {"price": 185,   "vol": 0.010, "tick": 0.01},
        "TSLA":    {"price": 250,   "vol": 0.030, "tick": 0.01},
        "DJI":     {"price": 39000, "vol": 0.007, "tick": 1.0},
        "CL":      {"price": 78,    "vol": 0.018, "tick": 0.01},
        "VIX":     {"price": 15,    "vol": 0.060, "tick": 0.01},
    }

    REGIMES = ["Uptrend", "Downtrend", "Sideways", "Volatile"]

    def __init__(self, asset: str = "SPX", seed: Optional[int] = None):
        self.asset = asset if asset in self.ASSETS else "SPX"
        info = self.ASSETS[self.asset]
        self.base_price = info["price"]
        self.base_vol = info["vol"]
        self.tick = info["tick"]

        if seed is not None:
            np.random.seed(seed)

        self.current_price = self.base_price
        self.regime = "Uptrend"
        self.regime_counter = 0
        self.regime_duration = np.random.randint(20, 60)
        self.history: List[Dict] = []
        self._dt = datetime.now() - timedelta(hours=500)

    def _next_regime(self):
        """Switch market regime"""
        self.regime = np.random.choice(
            self.REGIMES,
            p=[0.30, 0.25, 0.30, 0.15]
        )
        self.regime_duration = np.random.randint(15, 50)
        self.regime_counter = 0

    def _generate_bar(self) -> Dict:
        """Generate one OHLC bar based on current regime"""
        self.regime_counter += 1
        if self.regime_counter >= self.regime_duration:
            self._next_regime()

        vol = self.base_vol
        if self.regime == "Volatile":
            vol *= 2.5
        elif self.regime == "Sideways":
            vol *= 0.4

        # Trend drift
        if self.regime == "Uptrend":
            drift = self.base_vol * 0.3
        elif self.regime == "Downtrend":
            drift = -self.base_vol * 0.3
        else:
            drift = 0.0

        # Generate returns using log-normal model
        ret = drift + np.random.normal(0, vol)
        ret = np.clip(ret, -0.08, 0.08)

        open_price = self.current_price
        close_price = open_price * np.exp(ret)

        # Intrabar range
        bar_range = abs(np.random.normal(0, vol)) * open_price
        bar_range = max(bar_range, abs(close_price - open_price))

        if close_price > open_price:
            high = max(close_price, open_price) + bar_range * np.random.uniform(0.1, 0.6)
            low = min(close_price, open_price) - bar_range * np.random.uniform(0.05, 0.3)
        else:
            high = max(close_price, open_price) + bar_range * np.random.uniform(0.05, 0.3)
            low = min(close_price, open_price) - bar_range * np.random.uniform(0.1, 0.6)

        # Round to tick
        def snap(x):
            return round(round(x / self.tick) * self.tick, 6)

        open_p = snap(open_price)
        high_p = snap(max(open_price, close_price, high))
        low_p = snap(min(open_price, close_price, low))
        close_p = snap(close_price)

        # Ensure validity
        high_p = max(high_p, open_p, close_p)
        low_p = min(low_p, open_p, close_p)

        volume = int(np.random.lognormal(10, 0.8))
        self.current_price = close_p

        bar = {
            "timestamp": self._dt.isoformat(),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "regime": self.regime
        }
        self._dt += timedelta(minutes=15)
        return bar

    def generate_history(self, n_bars: int = 200) -> pd.DataFrame:
        """Generate historical OHLC bars"""
        bars = [self._generate_bar() for _ in range(n_bars)]
        self.history.extend(bars)
        df = pd.DataFrame(bars)
        return df

    def get_next_bar(self) -> Dict:
        """Generate the next live bar"""
        bar = self._generate_bar()
        self.history.append(bar)
        return bar

    def get_dataframe(self) -> pd.DataFrame:
        """Return current history as DataFrame"""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def get_current_price(self) -> float:
        return self.current_price


class AccuracyTracker:
    """
    Tracks prediction accuracy vs actual price movements.
    Implements the paper's evaluation metrics:
    - Directional accuracy α = C / T
    - Rate of return metrics: Rcc, Rmax, Rmin
    """

    def __init__(self):
        self.predictions: List[Dict] = []
        self.results: List[Dict] = []

    def record_prediction(self, prediction_id: str, direction: str,
                           entry_price: float, stop_loss: float,
                           take_profit: float, timestamp: float):
        """Record a new prediction for future validation"""
        self.predictions.append({
            "id": prediction_id,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": timestamp,
            "validated": False,
            "future_bars": []
        })

    def add_validation_bar(self, prediction_id: str, bar: Dict):
        """Add a future price bar to validate against"""
        for pred in self.predictions:
            if pred["id"] == prediction_id and not pred["validated"]:
                pred["future_bars"].append(bar)
                if len(pred["future_bars"]) >= 3:
                    self._validate(pred)

    def _validate(self, pred: Dict):
        """
        Validate prediction against next 3 bars.
        Per paper: for LONG, each bar closing above entry = correct hit (max 3)
        """
        pred["validated"] = True
        direction = pred["direction"]
        entry = pred["entry_price"]
        sl = pred["stop_loss"]
        tp = pred["take_profit"]
        bars = pred["future_bars"][:3]

        # Directional accuracy
        correct_hits = 0
        for bar in bars:
            close = bar["close"]
            if direction == "LONG" and close > entry:
                correct_hits += 1
            elif direction == "SHORT" and close < entry:
                correct_hits += 1

        # Rate of return calculations
        rcc = 0.0
        rmax = 0.0
        rmin = 0.0
        rho = 0.0005

        exited = False
        for bar in bars:
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]

            if direction == "LONG":
                # Track best/worst intrabar extremes
                rmax = max(rmax, (high - entry) / entry)
                rmin = min(rmin, (low - entry) / entry)
                # Check if TP or SL hit first
                if low <= sl:
                    rcc = (sl - entry) / entry   # stop-loss hit (negative)
                    exited = True
                    break
                elif high >= tp:
                    rcc = (tp - entry) / entry   # take-profit hit (positive)
                    exited = True
                    break
                else:
                    rcc = (close - entry) / entry

            else:  # SHORT
                rmax = max(rmax, (entry - low) / entry)
                rmin = min(rmin, (entry - high) / entry)
                if high >= sl:
                    rcc = (entry - sl) / entry   # stop-loss hit (negative for short)
                    exited = True
                    break
                elif low <= tp:
                    rcc = (entry - tp) / entry   # take-profit hit (positive for short)
                    exited = True
                    break
                else:
                    rcc = (entry - close) / entry

        result = {
            "id": pred["id"],
            "direction": direction,
            "correct_hits": correct_hits,
            "accuracy_score": correct_hits / 3.0,
            "correct": correct_hits >= 2,
            "rcc": round(rcc * 100, 4),
            "rmax": round(rmax * 100, 4),
            "rmin": round(rmin * 100, 4),
            "entry_price": entry,
            "actual_closes": [b["close"] for b in bars],
            "timestamp": pred["timestamp"]
        }
        self.results.append(result)
        pred["result"] = result

    def get_summary(self) -> Dict:
        """Get overall accuracy statistics"""
        if not self.results:
            return {
                "total_predictions": len(self.predictions),
                "total": len(self.predictions),
                "validated": 0,
                "accuracy": 0.0,
                "accuracy_pct": 0.0,
                "avg_rcc": 0.0,
                "avg_rmax": 0.0,
                "avg_rmin": 0.0,
                "long_accuracy": 0.0,
                "short_accuracy": 0.0,
                "results": []
            }

        validated = self.results
        total = len(validated)
        correct = sum(1 for r in validated if r["correct"])
        accuracy = correct / total if total > 0 else 0.0

        long_results = [r for r in validated if r["direction"] == "LONG"]
        short_results = [r for r in validated if r["direction"] == "SHORT"]

        long_acc = (sum(1 for r in long_results if r["correct"]) /
                    len(long_results)) if long_results else 0.0
        short_acc = (sum(1 for r in short_results if r["correct"]) /
                     len(short_results)) if short_results else 0.0

        return {
            "total_predictions": len(self.predictions),
            "total": len(self.predictions),          # frontend alias
            "validated": total,
            "accuracy": round(accuracy * 100, 1),
            "accuracy_pct": round(accuracy * 100, 1),  # frontend alias
            "avg_accuracy_score": round(np.mean([r["accuracy_score"] for r in validated]) * 100, 1),
            "avg_rcc": round(np.mean([r["rcc"] for r in validated]), 4),
            "avg_rmax": round(np.mean([r["rmax"] for r in validated]), 4),
            "avg_rmin": round(np.mean([r["rmin"] for r in validated]), 4),
            "long_accuracy": round(long_acc * 100, 1),
            "short_accuracy": round(short_acc * 100, 1),
            "results": validated[-20:]  # Last 20
        }
