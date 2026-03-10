import pandas_ta as ta

def indicator_agent(df):
    data = df.copy()

    # RSI
    data["RSI"] = ta.rsi(data["Close"], length=14)

    # MACD
    macd = ta.macd(data["Close"])
    data["MACD"] = macd["MACD_12_26_9"]
    data["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    # SAFE SMAs (NO pandas_ta)
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    data.dropna(inplace=True)
    latest = data.iloc[-1]

    rsi_signal = (
        "overbought" if latest["RSI"] > 70 else
        "oversold" if latest["RSI"] < 30 else
        "neutral"
    )

    macd_signal = "bullish" if latest["MACD"] > latest["MACD_SIGNAL"] else "bearish"
    trend_signal = "bullish" if latest["SMA_50"] > latest["SMA_200"] else "bearish"

    return {
        "RSI": round(latest["RSI"], 2),
        "RSI_signal": rsi_signal,
        "MACD_signal": macd_signal,
        "Trend_SMA": trend_signal
    }
