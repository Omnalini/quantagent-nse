"""
Nifty 50 Data Fetcher
Fetches real OHLC data for NSE-listed Nifty 50 stocks via yfinance.
Interval: 15m (up to 60 days of history available).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ── Nifty 50 constituents (NSE tickers) ───────────────────────────────────
NIFTY_50 = {
    "ADANIENT":    "Adani Enterprises",
    "ADANIPORTS":  "Adani Ports",
    "APOLLOHOSP":  "Apollo Hospitals",
    "ASIANPAINT":  "Asian Paints",
    "AXISBANK":    "Axis Bank",
    "BAJAJ-AUTO":  "Bajaj Auto",
    "BAJFINANCE":  "Bajaj Finance",
    "BAJAJFINSV":  "Bajaj Finserv",
    "BPCL":        "BPCL",
    "BHARTIARTL":  "Bharti Airtel",
    "BRITANNIA":   "Britannia",
    "CIPLA":       "Cipla",
    "COALINDIA":   "Coal India",
    "DIVISLAB":    "Divi's Labs",
    "DRREDDY":     "Dr. Reddy's",
    "EICHERMOT":   "Eicher Motors",
    "GRASIM":      "Grasim Industries",
    "HCLTECH":     "HCL Technologies",
    "HDFCBANK":    "HDFC Bank",
    "HDFCLIFE":    "HDFC Life",
    "HEROMOTOCO":  "Hero MotoCorp",
    "HINDALCO":    "Hindalco",
    "HINDUNILVR":  "Hindustan Unilever",
    "ICICIBANK":   "ICICI Bank",
    "ITC":         "ITC",
    "INDUSINDBK":  "IndusInd Bank",
    "INFY":        "Infosys",
    "JSWSTEEL":    "JSW Steel",
    "KOTAKBANK":   "Kotak Mahindra Bank",
    "LT":          "Larsen & Toubro",
    "M&M":         "Mahindra & Mahindra",
    "MARUTI":      "Maruti Suzuki",
    "NTPC":        "NTPC",
    "NESTLEIND":   "Nestle India",
    "ONGC":        "ONGC",
    "POWERGRID":   "Power Grid",
    "RELIANCE":    "Reliance Industries",
    "SBILIFE":     "SBI Life Insurance",
    "SHRIRAMFIN":  "Shriram Finance",
    "SBIN":        "State Bank of India",
    "SUNPHARMA":   "Sun Pharmaceuticals",
    "TCS":         "Tata Consultancy Services",
    "TATACONSUM":  "Tata Consumer Products",
    "TATAMOTORS":  "Tata Motors",
    "TATASTEEL":   "Tata Steel",
    "TECHM":       "Tech Mahindra",
    "TITAN":       "Titan Company",
    "ULTRACEMCO":  "UltraTech Cement",
    "UPL":         "UPL",
    "WIPRO":       "Wipro",
}

# ── NSE market hours (IST) ───────────────────────────────────────────────
NSE_OPEN_HOUR = 9
NSE_OPEN_MIN = 15
NSE_CLOSE_HOUR = 15
NSE_CLOSE_MIN = 30


def get_yf_ticker(symbol: str) -> str:
    """Convert bare NSE symbol to yfinance ticker (append .NS)."""
    s = symbol.upper().replace(".NS", "")
    return f"{s}.NS"


def fetch_ohlc(symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """
    Download OHLC data for a Nifty 50 stock from Yahoo Finance.

    Args:
        symbol:   NSE ticker without suffix, e.g. 'RELIANCE'
        interval: yfinance interval string ('1m','5m','15m','1h','1d', …)
        period:   yfinance period string ('1d','5d','1mo','3mo', …)
                  For 15m data yfinance provides up to 60 days.

    Returns:
        DataFrame with columns: open, high, low, close, volume, timestamp
        Sorted ascending by time, NaN rows dropped.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    ticker = get_yf_ticker(symbol)
    try:
        raw = yf.download(
            ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed for {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}. "
                         "Check symbol spelling and market hours.")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]

    # Keep only OHLCV
    needed = ["open", "high", "low", "close", "volume"]
    available = [c for c in needed if c in raw.columns]
    df = raw[available].copy()

    # Drop rows with any NaN in OHLC
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Ensure chronological order
    df.sort_index(inplace=True)

    # Add timestamp column (Unix seconds, float) for the frontend
    df["timestamp"] = df.index.astype(np.int64) // 10**9

    df.reset_index(drop=False, inplace=True)

    # Rename datetime index column
    dt_col = df.columns[0]
    df.rename(columns={dt_col: "datetime"}, inplace=True)

    return df


def get_latest_price(symbol: str) -> Optional[float]:
    """Return the most recent close price for a symbol."""
    try:
        df = fetch_ohlc(symbol, interval="1m", period="1d")
        if df.empty:
            return None
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def is_market_open() -> bool:
    """Return True if NSE is currently open (Mon–Fri 09:15–15:30 IST)."""
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:          # Saturday or Sunday
        return False
    open_t  = now.replace(hour=NSE_OPEN_HOUR,  minute=NSE_OPEN_MIN,  second=0, microsecond=0)
    close_t = now.replace(hour=NSE_CLOSE_HOUR, minute=NSE_CLOSE_MIN, second=0, microsecond=0)
    return open_t <= now <= close_t


def get_nifty_50_list() -> list[dict]:
    """Return list of {symbol, name} dicts for all Nifty 50 constituents."""
    return [{"symbol": sym, "name": name} for sym, name in NIFTY_50.items()]


# ── Simple in-memory cache to avoid redundant network calls ──────────────

_cache: dict = {}          # key → (timestamp_fetched, DataFrame)
_CACHE_TTL_SEC = 60 * 5   # 5 minutes


def fetch_ohlc_cached(symbol: str, interval: str = "15m",
                      period: str = "5d") -> pd.DataFrame:
    """fetch_ohlc with a 5-minute memory cache keyed by (symbol, interval, period)."""
    key = (symbol.upper(), interval, period)
    now = datetime.utcnow().timestamp()

    if key in _cache:
        cached_at, df = _cache[key]
        if now - cached_at < _CACHE_TTL_SEC:
            return df

    df = fetch_ohlc(symbol, interval, period)
    _cache[key] = (now, df)
    return df
