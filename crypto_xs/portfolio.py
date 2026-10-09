"""The book trades momentum-neutral forecasts in inverse-volatility decile legs, half to each model, with partial trading.

Each day each model's forecast is regressed across coins on the coin's ranked 30-day momentum and the fitted
part removed. Each model then goes long its top decile and short its bottom decile among tradable coins (a
perp return and a funding figure for the next day), weighted by 1/sigma within each leg, where sigma is the
20-day volatility floored at the day's 10th percentile; each leg sums to one. The two models' books are
averaged and the held book moves halfway from its drifted weights to that target each day. Cost is charged on
turnover against the drifted weights.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import COST, SAMPLES
from .models import load_scores, read_blocks

DECILE, MIN_COINS, TRADE_RATE = 0.10, 30, 0.5


def residualize(S, M, k=1.0):
    out = S.copy()
    for t in range(len(S)):
        ok = np.isfinite(S[t]) & np.isfinite(M[t])
        if ok.sum() < 10:
            continue
        m = M[t, ok] - M[t, ok].mean()
        s = S[t, ok] - S[t, ok].mean()
        out[t, ok] = S[t, ok] - k * ((m * s).sum() / (m * m).sum()) * m
    return out


def decile_legs(S, R, F, V):
    n_days, k = S.shape
    W = np.zeros((n_days, k))
    for t in range(n_days):
        ok = np.isfinite(S[t]) & np.isfinite(R[t]) & np.isfinite(F[t])
        n = int(ok.sum())
        if n < MIN_COINS:
            continue
        sig = V[t].astype(float).copy()
        sig[ok & ~np.isfinite(sig)] = np.nanmedian(sig[ok]) if np.isfinite(sig[ok]).any() else 1.0
        sig = np.maximum(sig, np.nanpercentile(sig[ok], 10))
        q = np.full(k, np.nan)
        q[ok] = rankdata(S[t][ok]) / n
        long_, short = ok & (q > 1 - DECILE), ok & (q <= DECILE)
        W[t, long_] = (1 / sig[long_]) / (1 / sig[long_]).sum()
        W[t, short] = -(1 / sig[short]) / (1 / sig[short]).sum()
    return W


def live_days(*scores, R, F):
    """Days a model's book is booked, which need at least 30 tradable coins and an earlier such day."""
    live = np.ones(len(R), bool)
    for S in scores:
        valid = (np.isfinite(S) & np.isfinite(R) & np.isfinite(F)).sum(1) >= MIN_COINS
        live &= valid & (np.cumsum(valid) > 1)
    return live


def run(T, live, R, F, cost, rate=TRADE_RATE):
    """Hold H_t = (1 - rate) D_t + rate T_t, where D_t is yesterday's book drifted by its returns; coins that
    stop being tradable are closed. Returns daily gross, turnover, net and the held weights."""
    Rz, Fz = np.nan_to_num(R), np.nan_to_num(F)
    tradable = np.isfinite(R) & np.isfinite(F)
    n, k = T.shape
    H = np.zeros_like(T); g = np.full(n, np.nan); turn = np.full(n, np.nan)
    prevH = np.zeros(k); prevg = 0.0; prevR = np.zeros(k); started = False
    for t in range(n):
        if not live[t]:
            if np.abs(T[t]).sum() > 0:                  # first day, the book forms and its return is not booked
                H[t] = T[t]; prevH = T[t].copy(); prevR = Rz[t]; prevg = T[t] @ Rz[t] - T[t] @ Fz[t]; started = True
            else:
                prevH = np.zeros(k); started = False
            continue
        D = prevH * (1 + prevR) / (1 + prevg) if started else prevH.copy()
        Dk = np.where(tradable[t], D, 0.0)
        Ht = (1 - rate) * Dk + rate * T[t] if started else T[t].copy()
        if started:
            g[t] = Ht @ Rz[t] - Ht @ Fz[t]
            turn[t] = np.abs(Ht - D).sum()
        H[t] = Ht; prevH = Ht; prevR = Rz[t]; prevg = Ht @ Rz[t] - Ht @ Fz[t]; started = True
    return dict(H=H, gross=g, turnover=turn, net=g - cost * turn)


def inputs_for(sample):
    """Wide day-by-coin arrays for one sample, holding raw scores, perp return, funding, spot return, volatility
    and ranked momentum. A missing perp return on a coin that has a score and funding takes the spot return."""
    a, b = SAMPLES[sample]
    sc = load_scores()
    sc = sc[(sc.index.get_level_values("day") >= a) & (sc.index.get_level_values("day") <= b)]
    Z, P = read_blocks("daily"), read_blocks("perps")
    idx = sc.index
    days = idx.get_level_values("day").unique().sort_values()
    cols = sorted(idx.get_level_values("sym").unique())
    wide = lambda s: s.reindex(idx).unstack().reindex(index=days, columns=cols).values
    X = dict(days=days, cols=cols, ridge=wide(sc.ridge), trees=wide(sc.trees), R=wide(P.perp_next), F=wide(P.fund_next),
             spot=wide(Z.next_ret), V=wide(Z.vol_raw), mom=wide(Z.mom_30d), wide=wide)
    fill = np.isfinite(X["ridge"]) & ~np.isfinite(X["R"]) & np.isfinite(X["F"]) & np.isfinite(X["spot"])
    X["R"][fill] = X["spot"][fill]
    X["spot_fills"] = int(fill.sum())
    return X


def book(sample):
    X = inputs_for(sample)
    Sr, St = residualize(X["ridge"], X["mom"]), residualize(X["trees"], X["mom"])
    T = 0.5 * decile_legs(Sr, X["R"], X["F"], X["V"]) + 0.5 * decile_legs(St, X["R"], X["F"], X["V"])
    b = run(T, live_days(Sr, St, R=X["R"], F=X["F"]), X["R"], X["F"], COST[sample])
    return X, b
