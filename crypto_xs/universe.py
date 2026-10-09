"""Monthly universe of the top 100 Binance spot USDT pairs by trailing 90-day quote volume, with an 80/120 buffer.

Membership for month M uses volume through the last day of M-1. A member stays while it ranks 120th or better
and still has data; a non-member enters when it ranks 80th or better; the list is trimmed or topped up to 100
by rank. Until 100 coins have a full 90-day history, the list is the plain top 100.
"""
import numpy as np
import pandas as pd

from .config import DATA, KLINES, eligible

WINDOW, ENTER, EXIT, SIZE = 90, 80, 120, 100
MONTHS = pd.period_range("2020-02", "2026-08", freq="M")


def daily_volume():
    """Daily quote volume per symbol from hourly spot bars; days without a bar count as zero."""
    cols = {}
    for f in sorted((KLINES / "spot").glob("*.parquet")):
        if not eligible(f.stem):
            continue
        d = pd.read_parquet(f, columns=["open_time", "volume", "quote_asset_volume", "taker_buy_base_volume"])
        d = d.drop_duplicates("open_time")
        d = d[d.taker_buy_base_volume <= d.volume]
        day = pd.to_datetime(d.open_time, unit="ms").dt.floor("D")
        cols[f.stem] = d.quote_asset_volume.groupby(day.values).sum()
    v = pd.DataFrame(cols)
    return v.reindex(pd.date_range(v.index.min(), v.index.max(), freq="D")).fillna(0.0)


def membership(vol):
    first = {s: vol.index[np.argmax(vol[s].values > 0)] for s in vol.columns}
    last_month = {s: vol.index[len(vol) - 1 - np.argmax(vol[s].values[::-1] > 0)].to_period("M") for s in vol.columns}
    rows, members, seeded = [], [], False
    for M in MONTHS:
        wend = (M - 1).end_time.normalize(); wstart = wend - pd.Timedelta(days=WINDOW - 1)
        total = vol.loc[(vol.index >= wstart) & (vol.index <= wend)].sum(axis=0); daily = total / WINDOW
        ranked = total[total > 0].sort_values(ascending=False)
        rank = {s: i + 1 for i, s in enumerate(ranked.index)}
        if not seeded:
            full = [s for s in ranked.index if first[s] <= wstart]
            if len(full) >= SIZE:
                members, seeded = full[:SIZE], True
            else:
                members = list(ranked.index[:SIZE])
        else:
            keep = [m for m in members if last_month[m] >= M - 1 and rank.get(m, 10**9) <= EXIT]
            kept = set(keep)
            members = keep + [s for s in ranked.index if s not in kept and rank[s] <= ENTER]
            if len(members) > SIZE:
                members = sorted(members, key=lambda s: rank.get(s, 10**9))[:SIZE]
            elif len(members) < SIZE:
                have = set(members)
                members = members + [s for s in ranked.index if s not in have][:SIZE - len(members)]
        for s in sorted(members, key=lambda s: rank.get(s, 10**9)):
            rows.append(dict(month=str(M), symbol=s, rank=rank.get(s), trailing_daily_volume=float(daily.get(s, 0.0))))
    return pd.DataFrame(rows)


def build():
    m = membership(daily_volume())
    m.to_csv(DATA / "universe.csv", index=False)
    print("universe:", m.month.nunique(), "months,", m.symbol.nunique(), "coins ever in it")
    return m


def load():
    return pd.read_csv(DATA / "universe.csv")
