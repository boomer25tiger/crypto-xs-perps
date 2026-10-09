"""Daily coin features, measured at 24:00 UTC when the book forms.

Each half-year block of decision days loads 120 days of hourly history before it, so every rolling window is
complete and blocks join without seams. A coin enters the panel on a decision day when it is in that month's
universe, has at least 20 hourly spot bars that day, defined order-flow features, a 20-day volatility and a
next-day spot return. Features are Gaussian-ranked across coins each day and missing values set to zero.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import inputs, universe
from .config import DATA_END, KLINES, NOT_TRADED, PANEL

FLOW = ["flow", "flow_persist", "flow_steady", "trade_size", "signed_volume", "late_flow"]
FEATURES = FLOW + ["mom_7d", "mom_30d", "max_ret_30d", "amihud", "btc_catchup", "basis", "flow_gap", "perp_share",
                   "ret_1d", "vol_20d", "log_volume", "funding", "asia_minus_us", "skew_168h", "volume_shock",
                   "dist_90d_high", "log_age", "down_beta", "spx_move", "funding_change", "funding_vs_basis"]
SQUEEZE = ["up_jumps_30d", "cascades_7d", "perp_flow_lead_3d", "perp_discount_3d", "neg_funding_9"]
BAR_COLS = ["open_time", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
            "taker_buy_quote_volume"]


def gauss_rank(s):
    """Rank within each day mapped to standard normal quantiles."""
    r = s.groupby(level="day").rank()
    n = s.groupby(level="day").transform("count")
    return pd.Series(norm.ppf((r - 0.5) / n), index=s.index)


def coins(d0, d1):
    m = universe.load()
    m = m[(m.month >= d0.strftime("%Y-%m")) & (m.month <= d1.strftime("%Y-%m")) & ~m.symbol.isin(NOT_TRADED)]
    return m, sorted(m.symbol.unique())


def hourly_matrices(market, syms, hours, lo, hi):
    P = {k: {} for k in "CQBN"}
    for s in syms:
        d = inputs.bars(market, s, lo, hi, BAR_COLS, clean_taker=True)
        if d is None:
            continue
        P["C"][s], P["Q"][s], P["B"][s] = d.close, d.quote_asset_volume, d.taker_buy_quote_volume
        P["N"][s] = d.number_of_trades.astype(float)
    return [pd.DataFrame(P[k]).reindex(index=hours, columns=syms) for k in "CQBN"]


def at_hour(X, hr, h):
    Y = X[hr == h]
    Y.index = Y.index.floor("D")
    return Y


def daily_panel(name, d0, d1):
    LS, LE = d0 - pd.Timedelta(days=120), min(d1 + pd.Timedelta(days=2), DATA_END)
    mem, syms = coins(d0, d1)
    hours = pd.date_range(LS, LE, freq="h", inclusive="left")
    C, Q, B, N = hourly_matrices("spot", syms, hours, LS, LE)
    Cu, Qu, Bu, _ = hourly_matrices("um", syms, hours, LS, LE)
    day = hours.floor("D"); hr = hours.hour
    Qd, Bd, Nd = (X.groupby(day).sum(min_count=1) for X in (Q, B, N))
    nbars = Q.notna().groupby(day).sum()
    Cd = at_hour(C, hr, 23)
    ret = Cd / Cd.shift(1) - 1
    S = {}

    # order flow, as taker-buy imbalance against its own 20-day mean
    late = hr >= 18
    Q6, B6 = Q[late].groupby(day[late]).sum(min_count=1), B[late].groupby(day[late]).sum(min_count=1)
    imb = (2 * Bd - Qd) / Qd.where(Qd > 0)
    xbar = imb.shift(1).rolling(20, min_periods=15).mean()
    dev = (2 * B - Q) / Q.where(Q > 0) - xbar.reindex(day).set_axis(hours)
    S["flow"] = imb - xbar
    S["flow_persist"] = np.sign(dev).groupby(day).mean()
    g = dev.groupby(day)
    S["flow_steady"] = g.mean() / g.std() * np.sqrt(dev.notna().groupby(day).sum())
    ats = Qd / Nd.where(Nd > 0)
    S["trade_size"] = np.sign(S["flow"]) * np.log(ats / ats.shift(1).rolling(20, min_periods=15).median())
    S["signed_volume"] = np.sign(S["flow"]) * np.log(Qd / Qd.shift(1).rolling(20, min_periods=15).mean())
    S["late_flow"] = (2 * B6 - Q6) / Q6.where(Q6 > 0) - xbar
    S["ret_1d"] = ret
    S["vol_20d"] = ret.rolling(20, min_periods=15).std()
    S["log_volume"] = np.log(Qd.where(Qd > 0))

    # price, liquidity, BTC lead-lag and the perpetual against spot
    Qud, Bud = Qu.groupby(day).sum(min_count=1), Bu.groupby(day).sum(min_count=1)
    Cud = at_hour(Cu, hr, 23)
    r6 = Cd / at_hour(C, hr, 17) - 1
    btc = ret["BTCUSDT"]
    mm = lambda x: x.rolling(60, min_periods=40).mean()
    beta = (mm(ret.mul(btc, axis=0)) - mm(ret).mul(mm(btc), axis=0)).div(mm(btc ** 2) - mm(btc) ** 2, axis=0)
    imb_u = (2 * Bud - Qud) / Qud.where(Qud > 0)
    share = np.log(Qud / (Qud + Qd))
    S.update({"mom_7d": Cd.shift(1) / Cd.shift(8) - 1, "mom_30d": Cd.shift(1) / Cd.shift(31) - 1,
              "max_ret_30d": ret.shift(1).rolling(30, min_periods=20).max(),
              "amihud": np.log((ret.abs() / Qd.where(Qd > 0)).rolling(20, min_periods=15).mean()),
              "btc_catchup": beta.mul(r6["BTCUSDT"], axis=0) - r6, "basis": np.log(Cud / Cd),
              "flow_gap": imb_u - imb, "perp_share": share - share.shift(1).rolling(20, min_periods=15).mean()})

    # sessions, return shape, attention, age, crash beta, equity spillover, funding
    lr = np.log(C).diff()
    a_, u_ = hr <= 7, (hr >= 13) & (hr <= 20)                              # Asia 00-08 UTC, US 13-21 UTC
    asia, us = lr[a_].groupby(day[a_]).sum(min_count=6), lr[u_].groupby(day[u_]).sum(min_count=6)
    S["asia_minus_us"] = (asia - us).rolling(7, min_periods=5).sum()
    S["skew_168h"] = at_hour(lr.rolling(168, min_periods=120).skew(), hr, 23)
    S["volume_shock"] = np.log(Qd / Qd.shift(1).rolling(20, min_periods=15).mean())
    S["dist_90d_high"] = np.log(Cd / Cd.rolling(90, min_periods=60).max())
    first = {}
    for s in syms:
        f = KLINES / "spot" / f"{s}.parquet"
        if f.exists():
            first[s] = pd.Timestamp(int(pd.read_parquet(f, columns=["open_time"]).open_time.min()), unit="ms", tz="UTC")
    A = pd.DataFrame({s: np.asarray((Cd.index - t0).days, float) for s, t0 in first.items()}, index=Cd.index).reindex(columns=syms)
    S["log_age"] = np.log(A.where(A > 0))
    down = np.broadcast_to((btc < 0).values[:, None], ret.shape)
    Rm = ret.where(down)
    Bm = pd.DataFrame(np.where(down, btc.values[:, None], np.nan), index=ret.index, columns=ret.columns).where(Rm.notna())
    Rm = Rm.where(Bm.notna())
    rl = lambda x: x.rolling(60, min_periods=20).mean()
    S["down_beta"] = (rl(Rm * Bm) - rl(Rm) * rl(Bm)) / (rl(Bm ** 2) - rl(Bm) ** 2)
    spx = inputs.sp500(LE).pct_change()
    td = spx.index[(spx.index >= Cd.index[0]) & (spx.index <= Cd.index[-1])]
    cr = at_hour(C, hr, 20).reindex(td)
    cr = cr / cr.shift(1) - 1                                               # coin return between S&P closes
    sr = spx.reindex(td)
    m6 = lambda x: x.rolling(60, min_periods=40).mean()
    bsp = (m6(cr.mul(sr, axis=0)) - m6(cr).mul(m6(sr), axis=0)).div(m6(sr ** 2) - m6(sr) ** 2, axis=0)
    spx_move = bsp.shift(1).mul(sr, axis=0).reindex(Cd.index)
    spx_move.loc[~Cd.index.isin(td)] = 0.0
    S["spx_move"] = spx_move
    fr = inputs.funding(LS, LE)
    fr = fr[fr.t.dt.hour == 0].copy()
    fr["funding_rate"] = fr.funding_rate * 8.0 / fr.interval                # 8-hour equivalent of the 00 UTC fix
    F0 = fr.pivot(index="t", columns="symbol", values="funding_rate")
    F0.index = F0.index.floor("D")
    F0 = F0.reindex(index=Cd.index, columns=syms)
    S["funding"] = F0.shift(-1)                                             # fixed at 00:00, when the book forms
    S["funding_change"] = F0.shift(-1) - F0
    nxt = ret.shift(-1)

    # eligibility and the long panel
    days = Cd.index[(Cd.index >= d0) & (Cd.index <= d1)]
    memb = pd.DataFrame(False, index=days, columns=syms); mstr = days.strftime("%Y-%m")
    for m, gm in mem.groupby("month"):
        memb.loc[mstr == m, [s for s in gm.symbol if s in syms]] = True
    clean = lambda X: X.reindex(index=days, columns=syms).replace([np.inf, -np.inf], np.nan)
    elig = memb & (nbars.reindex(index=days, columns=syms) >= 20) & clean(S["ret_1d"]).notna() & clean(S["vol_20d"]).notna()
    for k in FLOW:
        elig &= clean(S[k]).notna()
    keep = elig & clean(nxt).notna()
    r_, c_ = np.nonzero(keep.values)
    ix = pd.MultiIndex.from_arrays([days[r_], np.array(syms)[c_]], names=["day", "sym"])
    D = pd.DataFrame({k: clean(X).values[r_, c_] for k, X in S.items()}, index=ix)
    D["next_ret"] = clean(nxt).values[r_, c_]
    Z = pd.DataFrame(index=D.index)
    for k in FEATURES:
        if k != "funding_vs_basis":
            Z[k] = gauss_rank(D[k])
    both = D.funding.notna() & D.basis.notna()
    Z["funding_vs_basis"] = gauss_rank(gauss_rank(D.funding.where(both)) - gauss_rank(D.basis.where(both)))
    Z = Z.fillna(0.0)
    Z["next_ret"] = D.next_ret
    Z["vol_raw"] = D.vol_20d
    os.makedirs(PANEL, exist_ok=True)
    D.to_parquet(PANEL / f"{name}_raw.parquet")
    Z.to_parquet(PANEL / f"{name}_daily.parquet")
    print(f"{name}: daily panel {Z.shape[0]} coin-days, {len(days)} days", flush=True)
    return Z


def squeeze_features(name, d0, d1):
    """Short-squeeze and liquidation proxies from hourly bars for the coins in the block's daily panel. They count
    5% hourly up-jumps over 30 days, volume-spike crash hours over 7 days, perp minus spot taker imbalance over
    3 days, hours the perp traded below spot over 3 days, and negative funding prints among the last nine
    00/08/16 UTC fixes."""
    Z = pd.read_parquet(PANEL / f"{name}_daily.parquet", columns=["next_ret"])
    L0, L1 = d0 - pd.Timedelta(days=60), min(d1 + pd.Timedelta(days=2), DATA_END)
    cols = ["open_time", "close", "quote_asset_volume", "taker_buy_quote_volume"]

    def load(market, s):
        d = inputs.bars(market, s, L0, L1, cols)
        return None if d is None else d.drop(columns="open_time").asfreq("h")

    fr = inputs.funding(L0 - pd.Timedelta(days=10), L1)
    fr = fr[fr.t.dt.hour.isin([0, 8, 16])]
    out = []
    for s in sorted(set(Z.index.get_level_values("sym"))):
        sp = load("spot", s)
        if sp is None:
            continue
        um = load("um", s)
        lr = np.log(sp.close).diff()
        none = pd.Series(np.nan, index=sp.index)
        vol = (um.quote_asset_volume.reindex(sp.index) if um is not None else none).fillna(sp.quote_asset_volume)
        lrv = (np.log(um.close).diff().reindex(sp.index) if um is not None else none).fillna(lr)
        med = vol.shift(1).rolling(720, min_periods=360).median()
        f = pd.DataFrame(index=sp.index)
        f["up_jumps_30d"] = (lr > np.log(1.05)).astype(float).where(lr.notna()).rolling(720, min_periods=360).sum()
        f["cascades_7d"] = ((vol > 5 * med) & (lrv.abs() > 0.03)).astype(float).where(med.notna()).rolling(168, min_periods=84).sum()
        sq, sb = sp.quote_asset_volume.rolling(72, min_periods=48).sum(), sp.taker_buy_quote_volume.rolling(72, min_periods=48).sum()
        if um is not None:
            uq = um.quote_asset_volume.reindex(sp.index).rolling(72, min_periods=48).sum()
            ub = um.taker_buy_quote_volume.reindex(sp.index).rolling(72, min_periods=48).sum()
            f["perp_flow_lead_3d"] = (2 * ub - uq) / uq.where(uq > 0) - (2 * sb - sq) / sq.where(sq > 0)
            uc = um.close.reindex(sp.index)
            f["perp_discount_3d"] = (uc < sp.close).astype(float).where(uc.notna()).rolling(72, min_periods=48).sum()
        else:
            f["perp_flow_lead_3d"] = np.nan
            f["perp_discount_3d"] = np.nan
        d = f[f.index.hour == 23].copy()
        d.index = d.index.floor("D"); d.index.name = "day"
        g = fr[fr.symbol == s].set_index("t").funding_rate.sort_index()
        if len(g):
            neg = (g < 0).astype(float).rolling(9, min_periods=6).sum()
            d["neg_funding_9"] = neg.reindex(d.index + pd.Timedelta(days=1), method="ffill").values   # through the 00:00 fix
        else:
            d["neg_funding_9"] = np.nan
        d["sym"] = s
        out.append(d.reset_index())
    H = pd.concat(out).set_index(["day", "sym"]).sort_index().reindex(Z.index)
    H.to_parquet(PANEL / f"{name}_squeeze.parquet")
    print(f"{name}: squeeze features", H.shape, flush=True)
    return H
