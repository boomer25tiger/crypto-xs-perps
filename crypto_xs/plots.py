"""README figures from the saved books and summary, in a light and a dark version."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT, ROOT, SAMPLES, utc

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  design="#2a78d6", holdout="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", grid="#34342f",
                 design="#3987e5", holdout="#d95926"),
}


def style(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["muted"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(which="both", colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def equity_and_drawdown(t, suffix):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1], hspace=0.08))
    fig.patch.set_facecolor(t["surface"])
    start = 1.0
    for sample in SAMPLES:
        net = pd.read_parquet(OUTPUT / f"book_{sample}.parquet").net.dropna()
        eq = (1 + net).cumprod()
        ax.plot(eq.index, start * eq, color=t[sample], lw=1.6, label=f"{sample} sample")
        bx.fill_between(eq.index, 100 * (eq / eq.cummax() - 1), 0, color=t[sample], alpha=0.45, lw=0)
        ax.annotate(f"${start * eq.iloc[-1]:,.1f}", (eq.index[-1], start * eq.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=t["text"])
        start *= eq.iloc[-1]
    for a in (ax, bx):
        style(a, t)
        a.axvline(utc("2022-07-01"), color=t["muted"], lw=0.8)
    ax.text(utc("2022-07-20"), 1.05, "holdout opens 2022-07-01", color=t["muted"], fontsize=9, va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["$1", "$2", "$5", "$10", "$20", "$50"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylabel("growth of $1, net of cost (log scale)", color=t["muted"], fontsize=9)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    ax.set_title("Daily long-short book on Binance USDT perpetuals, after trading cost and funding",
                 color=t["text"], fontsize=11, loc="left", pad=10)
    bx.set_ylabel("drawdown within\nsample (%)", color=t["muted"], fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.92, bottom=0.07)
    fig.savefig(ROOT / "figures" / f"equity{suffix}.png", dpi=150, facecolor=t["surface"])
    plt.close(fig)


def by_period(t, suffix):
    s = json.load(open(OUTPUT / "summary.json"))
    rows = [(lab, sample, p) for sample in SAMPLES for lab, p in s[sample]["periods"].items()]
    x = np.arange(len(rows)); colors = [t[r[1]] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1], wspace=0.28))
    fig.patch.set_facecolor(t["surface"])
    ax, bx, cx = axes
    ax.bar(x, [r[2]["gross_bps"] for r in rows], width=0.78, color=colors, alpha=0.3)
    ax.bar(x, [r[2]["net_bps"] for r in rows], width=0.42, color=colors)
    for i, r in enumerate(rows):
        top = max(r[2]["net_bps"], r[2]["gross_bps"], 0)
        ax.annotate(f"t {r[2]['t_nw10']:.1f}", (i, top), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=t["muted"])
    ax.set_ylim(top=1.12 * max(max(r[2]["gross_bps"], r[2]["net_bps"]) for r in rows))
    ax.set_title("Net (narrow) and gross (wide), bps per day", color=t["text"], fontsize=10, loc="left")
    bx.bar(x, [r[2]["ic"] for r in rows], width=0.6, color=colors)
    bx.set_title("Mean daily rank IC", color=t["text"], fontsize=10, loc="left")
    cx.bar(x, [r[2]["dispersion_pct"] for r in rows], width=0.6, color=colors)
    cx.set_title("Cross-sectional dispersion, %", color=t["text"], fontsize=10, loc="left")
    for a in axes:
        style(a, t)
        a.axhline(0, color=t["muted"], lw=0.8)
        a.spines["bottom"].set_visible(False)
        a.set_xticks(x)
        a.set_xticklabels([r[0] for r in rows], rotation=45, ha="right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[k]) for k in SAMPLES]
    leg = fig.legend(handles, [f"{k} sample" for k in SAMPLES], loc="upper right", ncol=2, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.22)
    fig.savefig(ROOT / "figures" / f"by_period{suffix}.png", dpi=150, facecolor=t["surface"])
    plt.close(fig)


def run():
    (ROOT / "figures").mkdir(exist_ok=True)
    for name, t in THEMES.items():
        suffix = "" if name == "light" else "-dark"
        equity_and_drawdown(t, suffix)
        by_period(t, suffix)
