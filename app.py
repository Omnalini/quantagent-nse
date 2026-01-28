from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.data_loader import load_stock_data
from agents.indicator_agent import indicator_agent
from agents.trend_agent import trend_agent
from agents.pattern_agent import pattern_agent
from agents.decision_agent import decision_agent

import os

app = Flask(__name__)
CORS(app)

DATA_FOLDER = "data"

@app.route("/")
def home():
    return jsonify({"message": "Stock Advisory Backend is running"})


@app.route("/analyze", methods=["POST"])
def analyze_stock():
    """
    Input JSON:
    {
        "stock": "RELIANCE"
    }
    """

    data = request.get_json()
    stock = data.get("stock")

    if not stock:
        return jsonify({"error": "Stock name not provided"}), 400

    file_path = os.path.join(DATA_FOLDER, f"{stock}.csv")

    if not os.path.exists(file_path):
        return jsonify({"error": "Stock data not found"}), 404

    # Load data
    df = load_stock_data(file_path)

    # Run agents
    indicator = indicator_agent(df)
    trend = trend_agent(df)
    pattern = pattern_agent(df)

    current_price = df["Close"].iloc[-1]

    final_decision = decision_agent(
        indicator, trend, pattern, current_price
    )

    return jsonify({
        "stock": stock,
        "indicator": indicator,
        "trend": trend,
        "pattern": pattern,
        "final_decision": final_decision
    })


if __name__ == "__main__":
    app.run(debug=True)
