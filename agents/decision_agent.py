from agents.risk_agent import risk_agent

def decision_agent(indicator, trend, pattern, current_price):
    """
    Combines all agents and makes final decision.
    """

    rsi = indicator["RSI"]
    macd_signal = indicator["MACD_signal"]
    trend_dir = trend["trend"]
    pattern_type = pattern["pattern"]

    decision = "HOLD"
    reason = []

    # ---- BUY CONDITIONS ----
    if (
        trend_dir == "uptrend"
        and macd_signal == "bullish"
        and pattern_type in ["double_bottom", "breakout"]
        and rsi < 75
    ):
        decision = "BUY"
        reason.append("Uptrend confirmed")
        reason.append("Bullish MACD")
        reason.append(f"Bullish pattern: {pattern_type}")

    # ---- SELL CONDITIONS ----
    elif (
        trend_dir == "downtrend"
        and macd_signal == "bearish"
        and pattern_type in ["double_top", "breakdown"]
        and rsi > 25
    ):
        decision = "SELL"
        reason.append("Downtrend confirmed")
        reason.append("Bearish MACD")
        reason.append(f"Bearish pattern: {pattern_type}")

    else:
        reason.append("Signals are mixed or weak")

    # ---- RISK MANAGEMENT ----
    risk = None
    if decision in ["BUY", "SELL"]:
        risk = risk_agent(current_price, decision)

    return {
        "decision": decision,
        "reason": reason,
        "risk": risk
    }
