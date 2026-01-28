import pandas_ta as ta

def indicator_agent(df):
    data = df.copy()

    data["RSI"] = ta.rsi(data["Close"], length=14)
    macd = ta.macd(data["Close"])
    data["MACD"] = macd["MACD_12_26_9"]
    data["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    data["SMA_50"] = ta.sma(data["Close"], length=50)
    data["SMA_200"] = ta.sma(data["Close"], length=200)

    latest = data.iloc[-1]

    if latest["RSI"] > 70:
        rsi_signal = "overbought"
    elif latest["RSI"] < 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    macd_signal = "bullish" if latest["MACD"] > latest["MACD_SIGNAL"] else "bearish"
    trend_signal = "bullish" if latest["SMA_50"] > latest["SMA_200"] else "bearish"

    return {
        "RSI": round(latest["RSI"], 2),
        "RSI_signal": rsi_signal,
        "MACD_signal": macd_signal,
        "Trend_SMA": trend_signal
    }
