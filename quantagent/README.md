# QuantAgent — Multi-Agent LLM HFT System

Replication of the paper:  
**"QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading"**  
Xiong et al., 2025 (arXiv:2509.09995v3)

---

## Architecture

Four specialized agents + orchestrator:

```
OHLC Data
    │
    ├─► IndicatorAgent  (RSI, MACD, ROC, STOCH, WILLR)
    ├─► PatternAgent    (Double Bottom/Top, H&S, Triangles, Flags, Wedges...)
    ├─► TrendAgent      (OLS trendlines, support/resistance, Algorithm 1)
    └─► RiskAgent       (ρ=0.0005 stop-loss, r∈[1.2,1.8] take-profit, radar)
              │
              └─► DecisionAgent → LONG / SHORT + justification
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```

### 3. Open browser
```
http://localhost:5000
```

---

## Features

| Feature | Description |
|---|---|
| **IndicatorAgent** | RSI, MACD, ROC, Stochastic, Williams %R with contextual interpretation |
| **PatternAgent** | 18 canonical patterns (Double Bottom, Descending Triangle, Head & Shoulders, etc.) |
| **TrendAgent** | OLS regression on highs/lows → support/resistance channels (Algorithm 1) |
| **RiskAgent** | 6-dimension radar chart, stop-loss/take-profit zones (ρ = 0.0005) |
| **DecisionAgent** | LONG/SHORT with justification, trade setup, post-trade reflection |
| **Live Feed** | SSE-based real-time bar generation, auto-analysis every 10 bars |
| **Accuracy Tracker** | Directional accuracy α = C/T, Rcc/Rmax/Rmin per paper metrics |
| **Backtest** | Run N bars, analyze every M bars, report accuracy statistics |
| **Matplotlib Charts** | High-quality agent visualizations (Figures 4, 8, 13, 14 from paper) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/init` | Initialize simulation session |
| POST | `/api/analyze` | Run full 4-agent pipeline |
| POST | `/api/tick` | Advance one bar + validate predictions |
| GET  | `/api/stream` | SSE live feed (auto-analysis every 10 bars) |
| GET  | `/api/accuracy` | Accuracy statistics for session |
| GET  | `/api/history` | OHLC bar history |
| POST | `/api/backtest` | Run full backtest |
| POST | `/api/charts` | Generate matplotlib PNG charts |
| GET  | `/api/assets` | List available assets |

---

## Paper Metrics Implemented

- **Directional accuracy**: α = C/T (correct hits out of 3 bars)
- **Rate of Return**: Rcc (actual exit), Rmax (best case), Rmin (worst case)
- **Stop-loss**: ρ = 0.0005 (0.05% fixed, per Section 3.1.3)
- **Take-profit**: R = r·ρ, r ∈ [1.2, 1.8]
- **Algorithm 1**: OLS slope-aware trend detection with κ threshold τ

---

## Supported Assets

BTC/USD, NQ, SPX, ES, QQQ, AAPL, TSLA, DJI, CL, VIX

---

## Project Structure

```
quantagent/
├── app.py                    ← Flask server (all API endpoints)
├── requirements.txt
├── agents/
│   ├── indicators.py         ← IndicatorAgent: RSI/MACD/ROC/STOCH/WILLR
│   ├── patterns.py           ← PatternAgent: 18 chart pattern detectors
│   ├── trend.py              ← TrendAgent: Algorithm 1 OLS trendlines
│   ├── risk_decision.py      ← RiskAgent + DecisionAgent
│   └── orchestrator.py       ← QuantAgent pipeline coordinator
├── data/
│   └── simulator.py          ← MarketSimulator + AccuracyTracker
├── charts/
│   └── generator.py          ← Matplotlib chart generation (all 4 agents)
└── templates/
    └── index.html            ← Full dashboard UI (Chart.js + SSE)
```
