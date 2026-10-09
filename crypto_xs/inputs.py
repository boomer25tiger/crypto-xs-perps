"""Readers for the downloaded inputs."""
import pandas as pd

from .config import DATA, KLINES


def ms(t):
    return int(t.timestamp() * 1000)


def bars(market, symbol, lo, hi, cols, clean_taker=False):
    """Hourly bars with open time in [lo, hi), indexed by open time. Duplicate bars and bars off the hour are
    dropped; with clean_taker, so are bars whose taker-buy volume exceeds total volume."""
    f = KLINES / market / f"{symbol}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=cols, filters=[("open_time", ">=", ms(lo)), ("open_time", "<", ms(hi))])
    if d.empty:
        return None
    d = d.drop_duplicates("open_time")
    if clean_taker:
        d = d[d.taker_buy_base_volume <= d.volume * (1 + 1e-9)]
    t = pd.to_datetime(d.open_time, unit="ms", utc=True)
    on_hour = (t == t.dt.floor("h")).values
    d = d[on_hour]
    d.index = t[on_hour]
    return d


def funding(lo, hi):
    """Funding settlements with settlement time in [lo, hi), rounded to the hour. A missing interval means the
    standard 8-hour settlement."""
    fr = pd.read_csv(DATA / "funding.csv")
    fr["interval"] = pd.to_numeric(fr.funding_interval_hours, errors="coerce").fillna(8.0)
    fr["t"] = pd.to_datetime(fr.calc_time_ms, unit="ms", utc=True).dt.round("h")
    fr = fr[(fr.t >= lo) & (fr.t < hi)].drop_duplicates(["symbol", "t"])
    return fr[["symbol", "calc_time_ms", "funding_rate", "interval", "t"]].reset_index(drop=True)


def sp500(before):
    gs = pd.read_parquet(DATA / "sp500.parquet")
    gs["date"] = pd.to_datetime(gs.date).dt.tz_localize("UTC")
    gs = gs.set_index("date").adj_close
    return gs[gs.index < before]
