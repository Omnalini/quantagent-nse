#!/usr/bin/env python3
"""
Entry point for the QuantAgent dashboard.

    python run.py

Configuration is read from .env (see .env.example):
  GEMINI_API_KEY / ANTHROPIC_API_KEY — enables LLM enhancement
  SECRET_KEY                        — Flask session signing
  FLASK_DEBUG=1                     — Werkzeug debugger (local use only)
  PORT                              — defaults to 3000
"""

import os

from quantagent.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    print(f"QuantAgent Nifty 50 — starting on http://localhost:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port, threaded=True)
