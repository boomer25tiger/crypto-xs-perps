"""Next-day perpetual return and funding for each coin-day in the panel.

perp_next(d) is the perp close at 24:00 of d+1 over the close at 24:00 of d, minus 1. fund_next(d) is the sum
of every funding settlement in (d+1 00:00, d+2 00:00]. The day counts only when the closing 00 UTC fix exists
and the settlements cover at least 24 hours (sum of their interval hours), which also handles 4-hour, 1-hour
and emergency settlements. A perp whose bars stop for good exits at its last hourly close, with the funding
settled up to then.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import inputs
from .config import DATA_END, KLINES, PANEL


def last_bar(f):
    md = pq.ParquetFile(f).metadata
    i = md.schema.to_arrow_schema().get_field_index("open_time")
    return pd.Timestamp(max(md.row_group(r).column(i).statistics.max for r in range(md.num_row_groups)), unit="ms", tz="UTC")


def perp_returns(name, d0, d1):
    Z = pd.read_parquet(PANEL / f"{name}_daily.parquet", columns=["next_ret"])
    syms = sorted(set(Z.index.get_level_values("sym")))
    L0, L1 = d0 - pd.Timedelta(days=2), min(d1 + pd.Timedelta(days=3), DATA_END)
    days = pd.date_range(L0, L1 - pd.Timedelta(days=1), freq="D")
    closes, last = {}, {}
    for s in syms:
        f = KLINES / "um" / f"{s}.parquet"
        if not f.exists():
            continue
        tmax = last_bar(f)
        d = inputs.bars("um", s, L0, L1, ["open_time", "close"])
        if d is None:
            continue
        t = d.index.to_series()
        c23 = d[(t.dt.hour == 23).values]
        closes[s] = pd.Series(c23.close.values, index=c23.index.floor("D"))
        if tmax < DATA_END - pd.Timedelta(hours=2) and L0 <= tmax < L1:        # bars end for good inside this block
            last[s] = (tmax, float(d.close.values[np.argmax(t.values == tmax.to_datetime64())]) if (t == tmax).any() else np.nan)
    Cu = pd.DataFrame(closes).reindex(index=days, columns=syms)
    pr = Cu / Cu.shift(1) - 1
    fr = inputs.funding(L0, min(L1 + pd.Timedelta(days=1), DATA_END))
    fr["day"] = (fr.t - pd.Timedelta(hours=1)).dt.floor("D")                    # the 00:00 settlement of D+1 belongs to day D
    fs = fr.pivot_table(index="day", columns="symbol", values="funding_rate", aggfunc="sum").reindex(index=days, columns=syms)
    fi = fr.pivot_table(index="day", columns="symbol", values="interval", aggfunc="sum").reindex(index=days, columns=syms)
    h0 = fr[fr.t.dt.hour == 0].pivot_table(index="day", columns="symbol", values="funding_rate", aggfunc="count").reindex(index=days, columns=syms)
    held = fs.where((fi >= 24.0 - 1e-9) & (h0 >= 1))
    pn, fn = pr.shift(-1), held.shift(-1)                                        # decision day d holds day d+1
    for s, (tmax, px) in last.items():
        dd = tmax.floor("D") - pd.Timedelta(days=1)
        if dd in pn.index and not np.isfinite(pn.at[dd, s]) and np.isfinite(Cu.at[dd, s]) and np.isfinite(px):
            pn.at[dd, s] = px / Cu.at[dd, s] - 1
            g = fr[(fr.symbol == s) & (fr.t > tmax.floor("D")) & (fr.t <= tmax)]
            fn.at[dd, s] = float(g.funding_rate.sum())
    P = pd.concat({"perp_next": pn.stack(future_stack=True), "fund_next": fn.stack(future_stack=True)}, axis=1)
    P.index.names = ["day", "sym"]
    P = P.reindex(Z.index)
    P.to_parquet(PANEL / f"{name}_perps.parquet")
    print(f"{name}: perp returns, {len(last)} perps ending inside the block", flush=True)
    return P
