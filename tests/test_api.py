"""
Flask API tests. Every network call is mocked — nothing here touches yfinance
or an LLM provider. Cases marked T05/T08/T10 correspond to the report's test
table; the cross-symbol and lookahead cases are regression tests for defects
fixed during cleanup.
"""

import datetime as _dt

import numpy as np
import pytest
import pytz

from quantagent import app as app_module
from quantagent.agents import orchestrator as orch_module
from quantagent.data import nifty_fetcher
from conftest import make_ohlc


# ── Fakes ─────────────────────────────────────────────────────────────────

# Built by concatenation on purpose: these are placeholders, and writing the
# real prefixes as literals would trip every secret scanner pointed at the repo.
GEMINI_PREFIX    = "AI" + "za"
ANTHROPIC_PREFIX = "sk-" + "ant-"
FAKE_GEMINI_KEY  = GEMINI_PREFIX + "_placeholder_not_a_real_key"
FAKE_CLAUDE_KEY  = ANTHROPIC_PREFIX + "placeholder_not_a_real_key"


FAKE_LLM_RESPONSE = {
    "decision": "LONG",
    "justification": "Stubbed response.",
    "risk_reward_ratio": 1.5,
    "watch_for": "A close below support.",
    "predicted_close_price": 1500.0,
    "confirmed_pattern": "Double Bottom",
}


class FakeLLMClient:
    """Stands in for LLMClient so no request ever leaves the process."""

    payload = FAKE_LLM_RESPONSE

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.provider = "gemini" if api_key else "none"

    @property
    def available(self):
        return bool(self.api_key)

    def analyze_and_decide(self, **_kwargs):
        return dict(self.payload)


def _prices(base, n=80, drift=0.4, seed=1):
    rng = np.random.default_rng(seed)
    return base + np.cumsum(rng.normal(drift, base * 0.0015, n))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(orch_module, "LLMClient", FakeLLMClient)
    app_module.app.config.update(TESTING=True)
    # A fresh process-wide state per test: these dicts are module globals.
    app_module.trackers.clear()
    app_module.session_meta.clear()
    app_module.price_trackers.clear()
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture
def frames():
    """Two instruments at very different price levels, as in the reported bug."""
    return {
        "SBIN": make_ohlc(_prices(1062.80, seed=1)),
        "ITC":  make_ohlc(_prices(294.90, seed=2, drift=0.05)),
    }


def _serve(frames_by_symbol):
    """Build a fetch_ohlc_cached replacement backed by an in-memory dict."""
    def _fake(symbol, interval="15m", period="5d"):
        key = symbol.upper()
        if key not in frames_by_symbol:
            raise ValueError(f"No data returned for {key}")
        return frames_by_symbol[key]
    return _fake


# ── /api/analyze ──────────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_200_with_all_documented_fields(self, client, monkeypatch, frames):
        """Report test case T08."""
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        res = client.post("/api/analyze", json={
            "symbol": "SBIN", "interval": "15m", "period": "5d",
            "api_key": FAKE_GEMINI_KEY, "session_id": "s1",
        })
        assert res.status_code == 200
        d = res.get_json()

        for key in ("decision", "indicators", "pattern", "trend", "risk",
                    "ohlc", "predictions", "symbol", "prediction_id",
                    "session_id", "llm_state", "market_open"):
            assert key in d, f"missing top-level field: {key}"

        for key in ("direction", "entry_price", "stop_loss", "take_profit",
                    "confidence", "risk_reward_ratio", "justification"):
            assert key in d["decision"], f"missing decision field: {key}"

        assert d["decision"]["direction"] in ("LONG", "SHORT")
        assert 0.0 <= d["decision"]["confidence"] <= 1.0

    def test_response_echoes_the_resolved_symbol(self, client, monkeypatch, frames):
        """The frontend discards mismatches, so the server must always say which
        instrument it answered for."""
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        for symbol in ("SBIN", "ITC"):
            res = client.post("/api/analyze", json={
                "symbol": symbol, "api_key": FAKE_GEMINI_KEY, "session_id": "s1",
            })
            assert res.get_json()["symbol"] == symbol

    def test_fewer_than_30_bars_returns_400_with_a_useful_message(self, client, monkeypatch):
        short = make_ohlc(np.linspace(100, 110, 12))
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve({"TCS": short}))
        res = client.post("/api/analyze", json={"symbol": "TCS", "api_key": FAKE_GEMINI_KEY})
        assert res.status_code == 400
        error = res.get_json()["error"]
        assert "12" in error and "30" in error

    def test_missing_api_key_is_reported_not_faked(self, client, monkeypatch, frames):
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        res = client.post("/api/analyze", json={"symbol": "SBIN", "session_id": "s1"})
        assert res.status_code == 400
        assert res.get_json()["error"] == "no_api_key"

    def test_llm_state_is_one_of_three_values(self, client, monkeypatch, frames):
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        res = client.post("/api/analyze", json={
            "symbol": "SBIN", "api_key": FAKE_GEMINI_KEY, "session_id": "s1",
        })
        assert res.get_json()["llm_state"] in ("llm_off", "llm_error", "llm_ok")

    def test_empty_llm_response_reports_an_error_not_success(self, client, monkeypatch, frames):
        """An LLM call whose fields all come back null must not read as success."""
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        monkeypatch.setattr(FakeLLMClient, "payload", {})
        try:
            res = client.post("/api/analyze", json={
                "symbol": "SBIN", "api_key": FAKE_GEMINI_KEY, "session_id": "s1",
            })
            d = res.get_json()
            assert d["llm_state"] == "llm_error"
            assert d["llm_error"]
        finally:
            monkeypatch.setattr(FakeLLMClient, "payload", FAKE_LLM_RESPONSE)


# ── Trade levels (T05) ────────────────────────────────────────────────────

class TestTradeLevels:
    def test_long_and_short_bracket_the_entry_correctly(self, client, monkeypatch):
        """
        Report test case T05. Whichever direction comes out, the stop and target
        must sit on the correct sides of entry.
        """
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve({
            "UP":   make_ohlc(_prices(1500, drift=0.9, seed=5)),
            "DOWN": make_ohlc(_prices(1500, drift=-0.9, seed=6)),
        }))
        seen = set()
        for symbol in ("UP", "DOWN"):
            for direction in ("LONG", "SHORT"):
                monkeypatch.setattr(
                    FakeLLMClient, "payload", {**FAKE_LLM_RESPONSE, "decision": direction})
                d = client.post("/api/analyze", json={
                    "symbol": symbol, "api_key": FAKE_GEMINI_KEY,
                    "session_id": f"lv-{symbol}-{direction}",
                }).get_json()
                dec = d["decision"]
                entry, sl, tp = dec["entry_price"], dec["stop_loss"], dec["take_profit"]
                if dec["direction"] == "LONG":
                    assert sl < entry < tp, f"LONG levels out of order: {sl} {entry} {tp}"
                else:
                    assert tp < entry < sl, f"SHORT levels out of order: {tp} {entry} {sl}"
                seen.add(dec["direction"])
        monkeypatch.setattr(FakeLLMClient, "payload", FAKE_LLM_RESPONSE)
        assert seen == {"LONG", "SHORT"}, "both directions must be exercised"


# ── R² tracking: the 0.2 regression tests ─────────────────────────────────

class TestR2Tracking:
    def _analyze(self, client, symbol, session="r2"):
        return client.post("/api/analyze", json={
            "symbol": symbol, "api_key": FAKE_GEMINI_KEY, "session_id": session,
        }).get_json()

    def test_prediction_for_one_symbol_is_never_scored_against_another(
            self, client, monkeypatch, frames):
        """
        The reported defect: an OLS prediction of ₹294.59 made for ITC was scored
        against SBIN's ₹1062.80. Every recorded actual must lie in its own
        instrument's price range.
        """
        sbin_full, itc_full = frames["SBIN"], frames["ITC"]
        state = {"SBIN": sbin_full.iloc[:60], "ITC": itc_full.iloc[:60]}
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(state))

        self._analyze(client, "SBIN")          # pending for SBIN
        self._analyze(client, "ITC")           # different instrument
        state["SBIN"] = sbin_full.iloc[:61]    # a new SBIN bar closes
        state["ITC"] = itc_full.iloc[:61]
        self._analyze(client, "SBIN")          # commits the SBIN pending
        self._analyze(client, "ITC")

        for symbol, frame in (("SBIN", sbin_full), ("ITC", itc_full)):
            history = app_module.price_trackers[f"r2:{symbol}"]["history"]
            assert history, f"no samples recorded for {symbol}"
            lo, hi = frame["close"].min(), frame["close"].max()
            for row in history:
                assert row["symbol"] == symbol
                assert lo <= row["actual"] <= hi, (
                    f"{symbol} scored against an out-of-range price: {row['actual']}")

    def test_a_foreign_pending_record_is_discarded_not_scored(
            self, client, monkeypatch, frames):
        """Defence in depth: a pending record carrying another symbol is dropped."""
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        app_module.price_trackers["r2:SBIN"] = {
            "history": [],
            "pending": {"symbol": "ITC", "bar_key": "stale", "ols": 294.59, "llm": None},
            "scored_bars": [],
        }
        self._analyze(client, "SBIN")
        assert app_module.price_trackers["r2:SBIN"]["history"] == []

    def test_re_analysing_the_same_bar_does_not_duplicate_a_sample(
            self, client, monkeypatch, frames):
        """The identical duplicate rows in the bug report came from this path."""
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        for _ in range(4):
            self._analyze(client, "SBIN")
        assert app_module.price_trackers["r2:SBIN"]["history"] == []

    def test_r2_is_withheld_below_the_sample_floor(self, client, monkeypatch, frames):
        """A negative R² from three points is noise, not a result."""
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        self._analyze(client, "SBIN")
        d = client.get("/api/r2_scores?session_id=r2&symbol=SBIN").get_json()
        assert d["r2_ols"] is None
        assert d["r2_llm"] is None
        assert d["reason"] == "insufficient_samples"
        assert d["n_samples"] < d["min_samples"] == app_module.MIN_R2_SAMPLES

    def test_r2_is_reported_once_the_floor_is_reached(self, client, monkeypatch, frames):
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        history = [{"symbol": "SBIN", "bar_key": str(i), "ols": 1000.0 + i,
                    "llm": None, "actual": 1000.0 + i + 0.5}
                   for i in range(app_module.MIN_R2_SAMPLES)]
        app_module.price_trackers["r2:SBIN"] = {
            "history": history, "pending": None, "scored_bars": []}
        d = client.get("/api/r2_scores?session_id=r2&symbol=SBIN").get_json()
        assert d["reason"] is None
        assert d["r2_ols"] is not None


# ── /api/market_status (T10) ──────────────────────────────────────────────

class TestMarketStatus:
    @staticmethod
    def _freeze(monkeypatch, iso):
        ist = pytz.timezone("Asia/Kolkata")
        frozen = ist.localize(_dt.datetime.fromisoformat(iso))

        class _FrozenDatetime:
            @staticmethod
            def now(tz=None):
                return frozen

        monkeypatch.setattr(nifty_fetcher, "datetime", _FrozenDatetime)

    @pytest.mark.parametrize("moment,expected", [
        ("2026-01-05 10:00:00", True),   # Monday, mid-session
        ("2026-01-05 09:15:00", True),   # exactly at the open
        ("2026-01-05 15:30:00", True),   # exactly at the close
        ("2026-01-05 09:14:59", False),  # one second early
        ("2026-01-05 15:30:01", False),  # one second late
        ("2026-01-05 08:00:00", False),  # pre-market
        ("2026-01-03 10:00:00", False),  # Saturday
        ("2026-01-04 10:00:00", False),  # Sunday
    ])
    def test_open_for_a_pinned_ist_timestamp(self, client, monkeypatch, moment, expected):
        """Report test case T10."""
        self._freeze(monkeypatch, moment)
        d = client.get("/api/market_status").get_json()
        assert d["is_open"] is expected
        assert d["exchange"] == "NSE"


# ── /api/backtest ─────────────────────────────────────────────────────────

class TestBacktestWindowing:
    def test_no_lookahead_bias(self, client, monkeypatch):
        """
        The analysis window is df.iloc[:i] and the actual is df.iloc[i], so the
        bar being predicted is never visible to the agents that predict it.
        Each recorded entry price must equal the close of the bar BEFORE the
        actual — never the actual itself. This already holds; lock it in.
        """
        df = make_ohlc(_prices(1500, n=140, seed=9))
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve({"TECHM": df}))

        d = client.post("/api/backtest", json={
            "symbol": "TECHM", "interval": "5m", "period": "5d", "analysis_every": 3,
        }).get_json()
        assert d["n_signals"] > 0

        by_datetime = {str(v): i for i, v in enumerate(df["datetime"])}
        closes = df["close"].tolist()

        checked = 0
        for trade in d["sample_trades"]:
            idx = by_datetime.get(trade["datetime"])
            if idx is None:
                continue
            # The window ended at idx, so entry is that bar's close ...
            assert trade["entry"] == pytest.approx(round(closes[idx], 2)), \
                "entry must come from the last bar of the analysis window"
            # ... and the actual is the very next bar, which the window excluded.
            assert trade["actual_next"] == pytest.approx(round(closes[idx + 1], 2)), \
                "actual must be the bar immediately after the window"
            assert trade["entry"] != trade["actual_next"] or closes[idx] == closes[idx + 1]
            checked += 1
        assert checked >= 5, "expected several sample trades to verify"

    def test_reports_the_documented_summary_metrics(self, client, monkeypatch):
        df = make_ohlc(_prices(1500, n=140, seed=9))
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve({"TECHM": df}))
        d = client.post("/api/backtest", json={"symbol": "TECHM", "interval": "5m"}).get_json()
        for key in ("win_rate", "long_accuracy", "short_accuracy", "avg_profit",
                    "avg_loss", "profit_factor", "r2_ols", "n_ols_samples",
                    "equity_curve", "sample_trades"):
            assert key in d
        assert 0.0 <= d["win_rate"] <= 100.0
        assert len(d["equity_curve"]) == d["n_signals"]

    def test_too_few_bars_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "fetch_ohlc_cached",
                            _serve({"TCS": make_ohlc(_prices(1500, n=40))}))
        res = client.post("/api/backtest", json={"symbol": "TCS"})
        assert res.status_code == 400
        assert "50" in res.get_json()["error"]


# ── Misc endpoints ────────────────────────────────────────────────────────

class TestMiscEndpoints:
    def test_stocks_lists_the_nifty_50(self, client):
        stocks = client.get("/api/stocks").get_json()["stocks"]
        assert len(stocks) == 50
        assert {"symbol", "name"} <= set(stocks[0])

    def test_quote_echoes_its_symbol(self, client, monkeypatch, frames):
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        d = client.get("/api/quote?symbol=SBIN&interval=1m&period=1d&n=1").get_json()
        assert d["symbol"] == "SBIN"
        assert d["latest_price"] > 0

    def test_quote_for_an_unknown_symbol_returns_400(self, client, monkeypatch, frames):
        monkeypatch.setattr(app_module, "fetch_ohlc_cached", _serve(frames))
        res = client.get("/api/quote?symbol=NOSUCH")
        assert res.status_code == 400
        assert "error" in res.get_json()

    def test_set_api_key_rejects_an_unrecognised_format(self, client):
        res = client.post("/api/set_api_key", json={"api_key": "not-a-real-key"})
        assert res.status_code == 400

    @pytest.mark.parametrize("key,provider", [
        (FAKE_CLAUDE_KEY, "anthropic"),
        (FAKE_GEMINI_KEY, "gemini"),
    ])
    def test_set_api_key_detects_the_provider(self, client, key, provider):
        d = client.post("/api/set_api_key", json={"api_key": key}).get_json()
        assert d["provider"] == provider

    def test_llm_status_reports_llm_off_without_a_key(self, client, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        d = client.get("/api/llm_status").get_json()
        assert d["llm_state"] == "llm_off"
        assert d["llm_available"] is False
