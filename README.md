# QuantAgent-NSE

A five-agent LLM trading-analysis system for NSE / Nifty 50 equities: live OHLC ingestion,
algorithmic indicator–pattern–trend analysis, a vision-LLM review of the candlestick chart, and
a Flask dashboard with an offline backtester.

---

## Attribution

This project implements and extends **QuantAgent: Price-Driven Multi-Agent LLMs for
High-Frequency Trading** — Xiong et al., [arXiv:2509.09995v3](https://arxiv.org/abs/2509.09995).
The multi-agent decomposition, the fixed-ρ stop-loss design, the R = r·ρ take-profit rule, the
six-dimension risk radar and the LONG/SHORT-without-HOLD output are the paper's.

Built on top of that by this team:

- **NSE / Nifty 50 adaptation** — 50-constituent universe, IST market-hours handling, rupee-denominated levels.
- **Live data ingestion** via `yfinance`, replacing the paper's offline datasets.
- **A single-call `COMBINED_PROMPT` design** — all four agent narratives plus the final decision
  in one vision request, instead of one call per agent. Roughly a 4× cut in tokens and latency.
- **A Flask dashboard** — live chart, decision card, indicator/pattern/trend panels, risk radar.
- **An offline backtester** with equity curve, win rate, profit factor and OLS R².
- **An OLS-vs-LLM price-prediction comparison**, scored live and over the backtest.

This was a four-person final-year group project for the KIIT Application Development Lab.

---

## Team

| Contributor | Owned |
|---|---|
| Aryan Nalini | Orchestrator, Flask API, dashboard |
| _add name_ | IndicatorAgent, PatternAgent |
| _add name_ | TrendAgent, chart generation |
| _add name_ | RiskAgent, DecisionAgent, backtester |

<!-- Replace the placeholder rows with the real names and ownership split. -->

---

## Architecture

```
                       Nifty 50 OHLC  (yfinance, 1m–1h bars)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  IndicatorAgent            PatternAgent               TrendAgent
  RSI(14), MACD,            18 canonical chart         OLS regression on
  ROC, Stochastic,          patterns + pivot           pivot highs/lows →
  Williams %R               detection                  support/resistance
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                             RiskAgent
                 ρ = 0.05% stop-loss · take-profit R = r·ρ
                 six-dimension risk radar · r ∈ [1.2, 1.8]
                                  │
                                  ▼
                           DecisionAgent  ──►  LONG / SHORT
                 justification · trade setup · reflection
                                  │
                                  ▼
                  LLMClient  (Gemini 2.5 Flash / Claude Haiku 4.5)
        one vision call: candlestick PNG + all four agent reports
        → confirms or overturns the pattern, direction and R:R
```

The LLM is an enrichment layer, not a dependency: `QuantAgent.run(df, require_llm=False)` runs
the full algorithmic pipeline with no network access, which is how the backtester and the test
suite exercise it.

---

## Setup

```bash
git clone https://github.com/Omnalini/quantagent-nse.git
cd quantagent-nse

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add a GEMINI_API_KEY or ANTHROPIC_API_KEY
python run.py
```

Open <http://localhost:3000>. A key can also be pasted into the sidebar instead of `.env` — it is
kept in the server-side session and never written to disk.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google Gemini key (`AIza…`), free tier is sufficient |
| `ANTHROPIC_API_KEY` | Anthropic key (`sk-ant-…`) — either provider works |
| `SECRET_KEY` | Flask session signing. Without it, sessions reset on restart |
| `FLASK_DEBUG` | `1` enables the Werkzeug debugger. Leave at `0` — it allows remote code execution |
| `PORT` | Defaults to `3000` |

Run the tests with `pytest`. The whole suite is offline and takes a few seconds.

---

## Results

From the submitted report, Table 6.1 — backtest on **TECHM, 5-minute bars, 5-day window**,
re-analysed every 3 bars, rule-based only (no LLM):

| Metric | Value |
|---|---|
| Signals generated | 109 |
| Win rate | 55.0% |
| Long accuracy | 71.7% |
| Short accuracy | 42.9% |
| Profit factor | 1.38 |
| Average profit | ₹2.19 |
| Average loss | ₹1.94 |
| OLS R² (next-bar close) | 0.8708 |

Reproduce with **Quick Backtest** in the dashboard, or `POST /api/backtest` with
`{"symbol": "TECHM", "interval": "5m", "period": "5d", "analysis_every": 3}`. Figures depend on
the 5-day window `yfinance` returns on the day you run it.

Read the [Limitations](#limitations) before drawing conclusions from any of these numbers.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/stocks` | GET | Nifty 50 constituents |
| `/api/market_status` | GET | Whether NSE is open (09:15–15:30 IST, Mon–Fri) |
| `/api/quote` | GET | Latest bars and price for one symbol |
| `/api/analyze` | POST | Full five-agent pipeline on one symbol |
| `/api/backtest` | POST | Offline walk-forward backtest + equity curve |
| `/api/r2_scores` | GET | Live OLS-vs-LLM prediction R² |
| `/api/accuracy` | GET | Session directional accuracy |
| `/api/llm_status` | GET | `llm_off` / `llm_error` / `llm_ok` |
| `/api/set_api_key` | POST | Store a provider key in the session |

---

## Limitations

Read these before treating any figure above as evidence that the system works.

- **Single symbol, single timeframe, single period.** The reported backtest is TECHM on 5-minute
  bars over one 5-day window. Nothing is cross-validated across symbols, regimes or periods.
- **109 signals is a small sample.** A 55.0% win rate over 109 trades is not statistically
  separable from chance; the 95% interval comfortably spans 50%.
- **The long/short gap suggests drift, not skill.** Long accuracy (71.7%) far exceeds short
  (42.9%). That asymmetry is what you would expect from capturing market drift during a bullish
  window, rather than from a directional edge.
- **The R² is computed on price levels, not returns.** On a trending, autocorrelated series this
  is inflated: a persistence model predicting "next close = last close" scores nearly as well.
  R² on percentage returns would be close to zero. The 0.8708 mostly measures that prices are
  serially correlated.
- **No transaction costs.** Slippage, brokerage and bid-ask spread are not modelled. A 0.05%
  stop-loss sits *inside* a typical NSE round-trip cost, so realistic costs would consume most or
  all of the apparent edge.
- **Binary output by design.** LONG/SHORT with no HOLD is inherited from the paper, so genuinely
  ambiguous setups still produce a directional call.
- **Signal weights are a design choice**, not empirically fitted. The indicator/pattern/trend
  weights in `RiskAgent._determine_direction` were chosen by hand, not optimised.

### Known discrepancies with the submitted report

The academic report is submitted and frozen; these are recorded here rather than corrected there.

- **Stop-loss.** The report describes a 0.1% NSE-calibrated stop-loss as an original
  contribution. The code runs the paper's **0.05%** (`STOP_LOSS_RHO = 0.0005` in
  `quantagent/agents/risk_decision.py`), and every published figure was produced with 0.05%. The
  0.1% constant existed in the source but was referenced nowhere; it has been removed, and `rho`
  is now an explicit `RiskAgent` constructor argument defaulting to 0.0005.
- **Pattern count.** The report says seventeen patterns. `PATTERN_LIBRARY` holds nineteen
  entries: **18 detectable patterns** plus a `No Clear Pattern` fallback. The count in this
  README is the one the code implements.

---

## Project structure

```
quantagent-nse/
├── run.py                      entry point
├── requirements.txt
├── .env.example
├── quantagent/
│   ├── app.py                  Flask server, all API endpoints
│   ├── agents/
│   │   ├── indicators.py       RSI · MACD · ROC · Stochastic · Williams %R
│   │   ├── patterns.py         18 chart-pattern detectors + pivots
│   │   ├── trend.py            OLS trendlines, support/resistance (Algorithm 1)
│   │   ├── risk_decision.py    RiskAgent + DecisionAgent
│   │   ├── orchestrator.py     pipeline coordinator
│   │   └── llm_client.py       Gemini / Claude vision client
│   ├── data/
│   │   ├── nifty_fetcher.py    yfinance ingestion, cache, market hours
│   │   └── simulator.py        MarketSimulator + AccuracyTracker
│   ├── charts/generator.py     matplotlib agent visualisations
│   └── templates/index.html    dashboard
└── tests/                      offline test suite
```

---

## Disclaimer

This is an educational research replication. It is **not investment advice**, it routes no live
orders, and it has never traded real capital. The backtest is a walk-forward simulation over
historical bars with no transaction costs. Do not trade on its output.

## Licence

[MIT](LICENSE).
