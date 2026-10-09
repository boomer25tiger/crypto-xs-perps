"""README figures from the saved books and summary."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT, ROOT, SAMPLES, utc

COLORS = {"design": "#2f6690", "holdout": "#d1495b"}


def equity_and_drawdown():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    start = 1.0
    for sample in SAMPLES:
        net = pd.read_parquet(OUTPUT / f"book_{sample}.parquet").net.dropna()
        eq = (1 + net).cumprod()
        ax.plot(eq.index, start * eq, color=COLORS[sample], lw=1.4, label=f"{sample} sample")
        bx.fill_between(eq.index, 100 * (eq / eq.cummax() - 1), 0, color=COLORS[sample], alpha=0.35, lw=0)
        start *= eq.iloc[-1]
    for a in (ax, bx):
        a.axvline(utc("2022-07-01"), color="0.3", lw=0.8, ls="--")
        a.grid(alpha=0.25)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1, net of cost (log)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Daily long-short book on Binance USDT perpetuals, net of trading cost, funding included")
    bx.set_ylabel("drawdown within sample (%)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "equity.png", dpi=150)
    plt.close(fig)


def by_period():
    s = json.load(open(OUTPUT / "summary.json"))
    rows = [(lab, sample, p) for sample in SAMPLES for lab, p in s[sample]["periods"].items()]
    x = np.arange(len(rows)); colors = [COLORS[r[1]] for r in rows]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax.bar(x, [r[2]["gross_bps"] for r in rows], color=colors, alpha=0.3, label="gross")
    ax.bar(x, [r[2]["net_bps"] for r in rows], color=colors, width=0.5, label="net")
    for i, r in enumerate(rows):
        ax.annotate(f"t {r[2]['t_nw10']:.1f}", (i, max(r[2]["net_bps"], 0)), ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_ylabel("bps per day")
    ax.set_title("Net (narrow) and gross (wide) return by period")
    bx.bar(x, [r[2]["ic"] for r in rows], color=colors, width=0.5)
    bx.set_ylabel("mean daily rank IC")
    cx = bx.twinx()
    cx.plot(x, [r[2]["dispersion_pct"] for r in rows], color="0.2", marker="o", lw=1.2)
    cx.set_ylabel("cross-sectional dispersion (%)")
    bx.set_title("Forecast skill (bars) and return dispersion (line)")
    for a in (ax, bx):
        a.set_xticks(x)
        a.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=8)
        a.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "by_period.png", dpi=150)
    plt.close(fig)


def run():
    (ROOT / "figures").mkdir(exist_ok=True)
    equity_and_drawdown()
    by_period()
