def risk_agent(current_price, decision, risk_pct=0.02, reward_ratio=2):
    """
    Input:
        current_price: latest stock price
        decision: 'BUY' or 'SELL'
        risk_pct: percentage risk (default 2%)
        reward_ratio: reward multiplier (default 2x)

    Output:
        Dictionary with risk parameters
    """

    if decision == "BUY":
        stop_loss = current_price * (1 - risk_pct)
        target = current_price * (1 + risk_pct * reward_ratio)

    elif decision == "SELL":
        stop_loss = current_price * (1 + risk_pct)
        target = current_price * (1 - risk_pct * reward_ratio)

    else:
        return None

    return {
        "entry_price": round(current_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_reward": f"1:{reward_ratio}"
    }
