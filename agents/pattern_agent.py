import numpy as np

def pattern_agent(df, lookback=120, tolerance=0.02):
    """
    Detects basic chart patterns using swing highs/lows.
    
    tolerance: % difference allowed between peaks/lows
    """

    data = df.copy().tail(lookback)
    close = data["Close"].values

    swing_lows = []
    swing_highs = []

    # ---- FIND SWINGS ----
    for i in range(2, len(close) - 2):
        if close[i] < close[i - 1] and close[i] < close[i + 1]:
            swing_lows.append((i, close[i]))

        if close[i] > close[i - 1] and close[i] > close[i + 1]:
            swing_highs.append((i, close[i]))

    pattern = "none"
    confidence = 0.0

    # ---- DOUBLE BOTTOM ----
    if len(swing_lows) >= 2:
        low1, low2 = swing_lows[-2], swing_lows[-1]
        diff = abs(low1[1] - low2[1]) / low1[1]

        if diff < tolerance and close[-1] > low2[1]:
            pattern = "double_bottom"
            confidence = round(1 - diff, 2)

    # ---- DOUBLE TOP ----
    if len(swing_highs) >= 2:
        high1, high2 = swing_highs[-2], swing_highs[-1]
        diff = abs(high1[1] - high2[1]) / high1[1]

        if diff < tolerance and close[-1] < high2[1]:
            pattern = "double_top"
            confidence = round(1 - diff, 2)

    # ---- BREAKOUT / BREAKDOWN ----
    resistance = max(close[:-1])
    support = min(close[:-1])

    if close[-1] > resistance:
        pattern = "breakout"
        confidence = 0.8

    elif close[-1] < support:
        pattern = "breakdown"
        confidence = 0.8

    return {
        "pattern": pattern,
        "confidence": confidence,
        "swing_highs": len(swing_highs),
        "swing_lows": len(swing_lows)
    }
