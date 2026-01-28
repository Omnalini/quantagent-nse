import numpy as np

def trend_agent(df, lookback=60):
    """
    Input:
        df: pandas DataFrame with Close price and Date index
        lookback: number of recent days to assess slope
    Output:
        Dictionary with trend direction and strength
    """

    data = df.copy()

    # Moving averages (safe pandas)
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    data.dropna(inplace=True)

    # Latest values
    latest = data.iloc[-1]

    # ---- SLOPE CALCULATION ----
    recent_prices = data["Close"].tail(lookback).values
    x = np.arange(len(recent_prices))

    # Linear regression slope
    slope = np.polyfit(x, recent_prices, 1)[0]

    # ---- TREND LOGIC ----
    if latest["SMA_50"] > latest["SMA_200"] and slope > 0:
        trend = "uptrend"
    elif latest["SMA_50"] < latest["SMA_200"] and slope < 0:
        trend = "downtrend"
    else:
        trend = "sideways"

    # Trend strength (normalized slope)
    strength = round(abs(slope) / latest["Close"], 5)

    return {
        "trend": trend,
        "slope": round(slope, 2),
        "strength": strength
    }
