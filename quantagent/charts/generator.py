"""
Chart Generator: Produces matplotlib charts for PatternAgent and TrendAgent.
Replicates the paper's Figures 4, 7, 8, 13, 14.
Charts are rendered to base64 PNG for embedding in the web UI.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from typing import List, Tuple, Optional


# Dark theme matching the UI
DARK_BG = '#090c14'
CARD_BG = '#111827'
BORDER  = '#1e2d45'
TEXT    = '#e2e8f0'
MUTED   = '#64748b'
GREEN   = '#10b981'
RED     = '#ef4444'
BLUE    = '#3b82f6'
CYAN    = '#06b6d4'
YELLOW  = '#f59e0b'
PURPLE  = '#8b5cf6'

plt.rcParams.update({
    'figure.facecolor': CARD_BG,
    'axes.facecolor': DARK_BG,
    'axes.edgecolor': BORDER,
    'axes.labelcolor': MUTED,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': BORDER,
    'grid.linewidth': 0.5,
    'text.color': TEXT,
    'font.family': 'monospace',
    'font.size': 8,
})


def fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=CARD_BG, edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def draw_candlestick(ax, opens, highs, lows, closes, width=0.6):
    """Draw OHLC candlestick bars on a matplotlib axis"""
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        color = GREEN if c >= o else RED
        alpha = 0.85
        # Body
        body_h = abs(c - o)
        body_y = min(c, o)
        ax.bar(i, body_h, bottom=body_y, width=width,
               color=color, alpha=alpha, linewidth=0)
        # Wicks
        ax.plot([i, i], [l, min(c, o)], color=color, linewidth=0.8, alpha=alpha)
        ax.plot([i, i], [max(c, o), h], color=color, linewidth=0.8, alpha=alpha)


def generate_indicator_chart(ohlc_df: pd.DataFrame, title: str = "Indicator Agent") -> str:
    """
    Generate multi-panel indicator chart (Figure 14 equivalent).
    Shows RSI, MACD, Stochastic, ROC, Williams %R.
    """
    from agents.indicators import compute_rsi, compute_macd, compute_roc, compute_stochastic, compute_williams_r

    closes = ohlc_df['close'].values.astype(float)
    highs = ohlc_df['high'].values.astype(float)
    lows = ohlc_df['low'].values.astype(float)

    rsi = compute_rsi(closes)
    macd_line, signal_line, histogram = compute_macd(closes)
    roc = compute_roc(closes)
    k_series, d_series = compute_stochastic(highs, lows, closes)
    willr = compute_williams_r(highs, lows, closes)

    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=False)
    fig.suptitle(title, color=TEXT, fontsize=10, y=0.98)

    n = min(50, len(closes))

    # RSI
    ax = axes[0]
    rsi_n = rsi[-n:]
    ax.plot(rsi_n, color=BLUE, linewidth=1.2, label='RSI')
    ax.axhline(70, color=RED, linewidth=0.8, linestyle='--', alpha=0.6, label='OB=70')
    ax.axhline(30, color=GREEN, linewidth=0.8, linestyle='--', alpha=0.6, label='OS=30')
    ax.axhline(50, color=MUTED, linewidth=0.5, linestyle=':', alpha=0.4)
    ax.fill_between(range(len(rsi_n)), rsi_n, 50,
                    where=[r > 50 for r in rsi_n], alpha=0.07, color=GREEN)
    ax.fill_between(range(len(rsi_n)), rsi_n, 50,
                    where=[r < 50 for r in rsi_n], alpha=0.07, color=RED)
    ax.set_ylim(0, 100)
    ax.set_ylabel('RSI', color=MUTED, fontsize=7)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # MACD
    ax = axes[1]
    ml_n = macd_line[-n:]
    sl_n = signal_line[-n:]
    hist_n = histogram[-n:]
    x = np.arange(len(ml_n))
    ax.bar(x, hist_n, color=[GREEN if v >= 0 else RED for v in hist_n], alpha=0.6, width=0.8)
    ax.plot(ml_n, color=BLUE, linewidth=1.2, label='MACD')
    ax.plot(sl_n, color=RED, linewidth=1.0, linestyle='--', label='Signal')
    ax.axhline(0, color=MUTED, linewidth=0.5, alpha=0.5)
    ax.set_ylabel('MACD', color=MUTED, fontsize=7)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Stochastic
    ax = axes[2]
    k_n = k_series[-n:]
    d_n = d_series[-n:]
    ax.plot(k_n, color=CYAN, linewidth=1.2, label='%K')
    ax.plot(d_n, color=YELLOW, linewidth=1.0, linestyle='--', label='%D')
    ax.axhline(80, color=RED, linewidth=0.7, linestyle='--', alpha=0.5)
    ax.axhline(20, color=GREEN, linewidth=0.7, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Stoch', color=MUTED, fontsize=7)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # ROC
    ax = axes[3]
    roc_n = roc[-n:]
    ax.plot(roc_n, color=PURPLE, linewidth=1.2, label='ROC')
    ax.axhline(0, color=MUTED, linewidth=0.7, linestyle='-', alpha=0.5)
    ax.fill_between(range(len(roc_n)), roc_n, 0,
                    where=[r > 0 for r in roc_n], alpha=0.1, color=GREEN)
    ax.fill_between(range(len(roc_n)), roc_n, 0,
                    where=[r < 0 for r in roc_n], alpha=0.1, color=RED)
    ax.set_ylabel('ROC%', color=MUTED, fontsize=7)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # Williams %R
    ax = axes[4]
    wr_n = willr[-n:]
    ax.plot(wr_n, color=YELLOW, linewidth=1.2, label='Williams %R')
    ax.axhline(-20, color=RED, linewidth=0.7, linestyle='--', alpha=0.5, label='OB=-20')
    ax.axhline(-80, color=GREEN, linewidth=0.7, linestyle='--', alpha=0.5, label='OS=-80')
    ax.set_ylim(-100, 0)
    ax.set_ylabel('%R', color=MUTED, fontsize=7)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Bars (last 50)', color=MUTED, fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig_to_b64(fig)


def generate_pattern_chart(ohlc_df: pd.DataFrame, pattern_name: str,
                            pivot_highs: List[Tuple], pivot_lows: List[Tuple],
                            title: str = "Pattern Agent") -> str:
    """
    Generate annotated candlestick chart for PatternAgent (Figure 4 equivalent).
    Shows detected pattern with pivot overlays.
    """
    n = min(60, len(ohlc_df))
    df = ohlc_df.tail(n).reset_index(drop=True)

    opens  = df['open'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f'{title} — {pattern_name}', color=TEXT, fontsize=10)

    # Draw candlesticks
    draw_candlestick(ax, opens, highs, lows, closes)

    # Overlay pattern context
    offset = len(ohlc_df) - n

    # Draw pivot highs
    for idx, price in pivot_highs:
        adj = idx - offset
        if 0 <= adj < n:
            ax.scatter(adj, price, color=RED, s=30, zorder=5, marker='v', alpha=0.8)
            ax.annotate('H', (adj, price), textcoords='offset points',
                        xytext=(0, 6), color=RED, fontsize=6, ha='center')

    # Draw pivot lows
    for idx, price in pivot_lows:
        adj = idx - offset
        if 0 <= adj < n:
            ax.scatter(adj, price, color=GREEN, s=30, zorder=5, marker='^', alpha=0.8)
            ax.annotate('L', (adj, price), textcoords='offset points',
                        xytext=(0, -10), color=GREEN, fontsize=6, ha='center')

    # Draw connecting lines between pivots (pattern geometry)
    if len(pivot_highs) >= 2:
        for i in range(len(pivot_highs) - 1):
            x1, y1 = pivot_highs[i][0] - offset, pivot_highs[i][1]
            x2, y2 = pivot_highs[i+1][0] - offset, pivot_highs[i+1][1]
            if 0 <= x1 < n and 0 <= x2 < n:
                ax.plot([x1, x2], [y1, y2], color=RED, linewidth=1.2,
                        linestyle='--', alpha=0.6)

    if len(pivot_lows) >= 2:
        for i in range(len(pivot_lows) - 1):
            x1, y1 = pivot_lows[i][0] - offset, pivot_lows[i][1]
            x2, y2 = pivot_lows[i+1][0] - offset, pivot_lows[i+1][1]
            if 0 <= x1 < n and 0 <= x2 < n:
                ax.plot([x1, x2], [y1, y2], color=BLUE, linewidth=1.2,
                        linestyle='--', alpha=0.6)

    # Pattern label
    mid_x = n // 2
    price_range = highs.max() - lows.min()
    label_y = lows.min() + price_range * 0.1
    ax.text(mid_x, label_y, pattern_name, color=CYAN, fontsize=9,
            ha='center', alpha=0.7, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD_BG, alpha=0.5))

    ax.set_xlabel('Bars', color=MUTED, fontsize=7)
    ax.set_ylabel('Price', color=MUTED, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)

    # Legend
    h_patch = mpatches.Patch(color=RED, label='Resistance pivots', alpha=0.7)
    l_patch = mpatches.Patch(color=GREEN, label='Support pivots', alpha=0.7)
    ax.legend(handles=[h_patch, l_patch], fontsize=7, loc='upper left',
              framealpha=0.3)

    plt.tight_layout()
    return fig_to_b64(fig)


def generate_trend_chart(ohlc_df: pd.DataFrame,
                          resist_slope: float, resist_intercept: float,
                          support_slope: float, support_intercept: float,
                          trend: str, title: str = "Trend Agent") -> str:
    """
    Generate annotated K-line chart with support/resistance trendlines.
    Replicates paper Figures 8 and 13 (TrendAgent visualizations).
    """
    n = min(60, len(ohlc_df))
    df = ohlc_df.tail(n).reset_index(drop=True)

    opens  = df['open'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values

    fig, ax = plt.subplots(figsize=(11, 5))
    trend_color = GREEN if trend == 'Uptrend' else (RED if trend == 'Downtrend' else YELLOW)
    fig.suptitle(f'{title} — {trend}', color=trend_color, fontsize=10)

    # Candlesticks
    draw_candlestick(ax, opens, highs, lows, closes)

    # Trendlines (computed over window N)
    window = min(20, n)
    x_trend = np.arange(window, dtype=float)
    resist_y = resist_slope * x_trend + resist_intercept
    support_y = support_slope * x_trend + support_intercept

    # Align to end of chart
    x_plot = np.arange(n - window, n, dtype=float)
    ax.plot(x_plot, resist_y, color=RED, linewidth=1.5, label='Resistance', alpha=0.85)
    ax.plot(x_plot, support_y, color=BLUE, linewidth=1.5, label='Support', alpha=0.85)

    # Fill channel
    ax.fill_between(x_plot, support_y, resist_y, alpha=0.04,
                    color=CYAN if trend == 'Uptrend' else RED)

    # Annotate last values
    last_r = resist_y[-1]
    last_s = support_y[-1]
    ax.annotate(f'R: {last_r:.2f}', xy=(n - 1, last_r),
                xytext=(4, 0), textcoords='offset points',
                color=RED, fontsize=7, va='center')
    ax.annotate(f'S: {last_s:.2f}', xy=(n - 1, last_s),
                xytext=(4, 0), textcoords='offset points',
                color=BLUE, fontsize=7, va='center')

    # Direction arrow
    arrow_x = n - 5
    arrow_dy = (last_r - last_s) * 0.3
    arrow_color = trend_color
    if trend == 'Uptrend':
        ax.annotate('', xy=(arrow_x, last_s + arrow_dy),
                    xytext=(arrow_x, last_s),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    elif trend == 'Downtrend':
        ax.annotate('', xy=(arrow_x, last_r - arrow_dy),
                    xytext=(arrow_x, last_r),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

    ax.set_xlabel('Bars', color=MUTED, fontsize=7)
    ax.set_ylabel('Price', color=MUTED, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.3)

    plt.tight_layout()
    return fig_to_b64(fig)


def generate_risk_chart(ohlc_df: pd.DataFrame,
                         entry: float, stop_loss: float, take_profit: float,
                         direction: str, rr_ratio: float,
                         title: str = "Risk Agent") -> str:
    """
    Generate risk-reward zone chart (Figure 2 equivalent).
    Shows entry, stop-loss, take-profit levels on price chart.
    """
    n = min(50, len(ohlc_df))
    df = ohlc_df.tail(n).reset_index(drop=True)
    closes = df['close'].values

    fig, ax = plt.subplots(figsize=(10, 4))
    dir_color = GREEN if direction == 'LONG' else RED
    fig.suptitle(f'{title} — {direction} | R:R = 1:{rr_ratio:.2f}',
                 color=dir_color, fontsize=10)

    # Price line
    ax.plot(closes, color=TEXT, linewidth=1.2, alpha=0.8, label='Price')

    # Horizontal levels
    ax.axhline(entry, color=CYAN, linewidth=1.5, linestyle='-', alpha=0.9, label=f'Entry: {entry:.4f}')
    ax.axhline(stop_loss, color=RED, linewidth=1.5, linestyle='--', alpha=0.8, label=f'Stop: {stop_loss:.4f}')
    ax.axhline(take_profit, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.8, label=f'Target: {take_profit:.4f}')

    # Fill zones
    ax.fill_between(range(n), entry, take_profit, alpha=0.06, color=GREEN, label='Reward zone')
    ax.fill_between(range(n), stop_loss, entry, alpha=0.06, color=RED, label='Risk zone')

    # Annotations on right
    xr = n - 1
    ax.annotate(f'TP: {take_profit:.4f}', xy=(xr, take_profit),
                xytext=(4, 0), textcoords='offset points',
                color=GREEN, fontsize=7, va='center')
    ax.annotate(f'SL: {stop_loss:.4f}', xy=(xr, stop_loss),
                xytext=(4, 0), textcoords='offset points',
                color=RED, fontsize=7, va='center')
    ax.annotate(f'Entry: {entry:.4f}', xy=(xr, entry),
                xytext=(4, 0), textcoords='offset points',
                color=CYAN, fontsize=7, va='center')

    ax.set_xlabel('Bars', color=MUTED, fontsize=7)
    ax.set_ylabel('Price', color=MUTED, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.3, ncol=2)

    plt.tight_layout()
    return fig_to_b64(fig)


def generate_all_charts(ohlc_df: pd.DataFrame, analysis_result: dict) -> dict:
    """
    Generate all four agent charts from a complete analysis result.
    Returns dict of base64 PNG images.
    """
    charts = {}

    try:
        charts['indicator'] = generate_indicator_chart(ohlc_df, "Indicator Agent")
    except Exception as e:
        charts['indicator'] = None

    try:
        pat = analysis_result['pattern']
        charts['pattern'] = generate_pattern_chart(
            ohlc_df,
            pat['name'],
            pat.get('pivot_points', [])[:4],
            pat.get('pivot_points', [])[-4:],
            "Pattern Agent"
        )
    except Exception as e:
        charts['pattern'] = None

    try:
        tre = analysis_result['trend']
        charts['trend'] = generate_trend_chart(
            ohlc_df,
            tre['resistance']['slope'],
            tre['resistance']['intercept'],
            tre['support']['slope'],
            tre['support']['intercept'],
            tre['classification'],
            "Trend Agent"
        )
    except Exception as e:
        charts['trend'] = None

    try:
        d = analysis_result['decision']
        charts['risk'] = generate_risk_chart(
            ohlc_df,
            d['entry_price'], d['stop_loss'], d['take_profit'],
            d['direction'], d['risk_reward_ratio'],
            "Risk Agent"
        )
    except Exception as e:
        charts['risk'] = None

    return charts
