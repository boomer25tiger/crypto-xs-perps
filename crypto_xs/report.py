"""Performance summary for each sample and the holdout test.

The holdout test is one-sided: the Newey-West t statistic (10 lags) of daily net returns must reach the hurdle
fixed before the read.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr

from . import portfolio
from .config import COST, OUTPUT, SAMPLES, TRIAL_CORRELATION, TRIALS, utc
from .models import read_blocks

EULER = 0.5772156649015329
PERIODS = {"design": [("2021 H1", "2021-01-01", "2021-06-30"), ("2021 H2", "2021-07-01", "2021-12-31"),
                      ("2022 H1", "2022-01-01", "2022-06-30")],
           "holdout": [("2022 H2", "2022-07-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
                       ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
                       ("2026 Jan-Aug", "2026-01-01", "2026-08-31")]}


def nw_t(x, lag):
    x = np.asarray(x); e = x - x.mean(); s = (e * e).mean()
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] * e[:-k]).mean()
    return float(x.mean() / np.sqrt(s / len(x)))


def bootstrap_mean(x, block=10, reps=10000, seed=0):
    """Circular block bootstrap 95% interval for the mean."""
    x = np.asarray(x); n = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); m = np.empty(reps)
    for r in range(reps):
        s = rng.integers(0, n, nb)
        m[r] = x[(s[:, None] + np.arange(block)[None, :]).ravel()[:n] % n].mean()
    return np.percentile(m, [2.5, 97.5])


def hurdle(n=TRIALS, rho=TRIAL_CORRELATION):
    """max(2, sqrt(1 - rho) E[max of n standard normals]) (Bailey and Lopez de Prado, 2014)."""
    emax = (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))
    return max(2.0, float(np.sqrt(1 - rho) * emax))


def zrow(S):
    with np.errstate(all="ignore"):
        return (S - np.nanmean(S, 1, keepdims=True)) / np.nanstd(S, 1, keepdims=True)


def summarize(sample, X, b):
    days = X["days"]; R, F, H = X["R"], X["F"], b["H"]
    net = pd.Series(b["net"], index=days).dropna()
    gross = pd.Series(b["gross"], index=days).reindex(net.index)
    turn = pd.Series(b["turnover"], index=days).reindex(net.index)
    on = lambda a: pd.Series(a, index=days).reindex(net.index)
    tradable = np.isfinite(R) & np.isfinite(F)
    Rz, Fz = np.nan_to_num(R), np.nan_to_num(F)
    pos, neg = np.where(H > 0, H, 0), np.where(H < 0, H, 0)
    long_leg = on((pos * Rz).sum(1) - (pos * Fz).sum(1)); short_leg = on((neg * Rz).sum(1) - (neg * Fz).sum(1))
    funding = on(-(H * Fz).sum(1))
    disp = on(np.nanstd(np.where(tradable, R, np.nan), 1))
    market = on(np.nanmean(np.where(tradable, R, np.nan), 1))
    btc = on(R[:, X["cols"].index("BTCUSDT")])
    combo = 0.5 * (zrow(X["ridge"]) + zrow(X["trees"]))
    ic = []
    for t in range(len(days)):
        m = np.isfinite(X["ridge"][t]) & np.isfinite(X["trees"][t]) & tradable[t]
        ic.append(spearmanr(combo[t][m], R[t][m])[0] if m.sum() >= 30 else np.nan)
    ic = on(ic)
    eq = (1 + net).cumprod(); dd = eq / eq.cummax() - 1
    trough = dd.idxmin(); peak = eq[:trough].idxmax(); rec = eq[trough:][eq[trough:] >= eq[peak]]
    adv = np.exp(X["wide"](read_blocks("raw").log_volume))
    dH = np.abs(np.diff(H, axis=0, prepend=np.zeros((1, H.shape[1]))))
    traded = (dH > 1e-9) & np.isfinite(adv) & (adv > 0)
    share = dH[traded] / adv[traded]
    lo, hi = bootstrap_mean(net.values)
    out = dict(first=str(net.index[0].date()), last=str(net.index[-1].date()), days=len(net),
               cost_bps=round(COST[sample] * 1e4, 2), net_bps=round(net.mean() * 1e4, 2), t_nw10=round(nw_t(net, 10), 3),
               t_nw5=round(nw_t(net, 5), 3), gross_bps=round(gross.mean() * 1e4, 2), turnover=round(turn.mean(), 4),
               sharpe=round(net.mean() / net.std() * np.sqrt(365), 2), skew=round(skew(net), 2), excess_kurtosis=round(kurtosis(net), 2),
               boot95_bps=[round(lo * 1e4, 1), round(hi * 1e4, 1)], end_value_per_10k=round(float(1e4 * eq.iloc[-1])),
               max_drawdown_pct=round(dd.min() * 100, 1), peak=str(peak.date()), trough=str(trough.date()),
               recovered=str(rec.index[0].date()) if len(rec) else None,
               long_leg_bps=round(long_leg.mean() * 1e4, 2), short_leg_bps=round(short_leg.mean() * 1e4, 2),
               funding_bps=round(funding.mean() * 1e4, 2), ic=round(float(ic.mean()), 4), ic_t_nw10=round(nw_t(ic.dropna(), 10), 2),
               dispersion_pct=round(float(disp.mean()) * 100, 2), corr_market=round(net.corr(market), 3), corr_btc=round(net.corr(btc), 3),
               worst_day_bps=round(net.min() * 1e4, 1), best_day_bps=round(net.max() * 1e4, 1),
               worst5_mean_bps=round(net[net <= net.quantile(0.05)].mean() * 1e4, 1),
               account_usd_p95_trade_1pct_volume=float(np.round(0.01 / np.quantile(share, 0.95), -4)),
               tradable_coins_median=int(np.median(tradable.sum(1))), held_coins_median=int(np.median((np.abs(H) > 1e-9).sum(1))),
               spot_fills=X["spot_fills"], periods={})
    for label, a, z in PERIODS[sample]:
        k = (net.index >= utc(a)) & (net.index <= utc(z)); x = net[k]
        out["periods"][label] = dict(days=int(k.sum()), net_bps=round(x.mean() * 1e4, 2), t_nw10=round(nw_t(x, 10), 2),
                                     sharpe=round(x.mean() / x.std() * np.sqrt(365), 2), gross_bps=round(gross[k].mean() * 1e4, 2),
                                     funding_bps=round(funding[k].mean() * 1e4, 2), ic=round(float(ic[k].mean()), 4),
                                     dispersion_pct=round(float(disp[k].mean()) * 100, 2))
    return out


def run():
    summary = {}
    for sample in SAMPLES:
        X, b = portfolio.book(sample)
        pd.DataFrame({"net": b["net"], "gross": b["gross"], "turnover": b["turnover"]}, index=X["days"]).to_parquet(OUTPUT / f"book_{sample}.parquet")
        summary[sample] = summarize(sample, X, b)
    h = hurdle()
    summary["holdout_test"] = dict(trials=TRIALS, trial_correlation=TRIAL_CORRELATION, hurdle=round(h, 2),
                                   t_nw10=summary["holdout"]["t_nw10"], passed=summary["holdout"]["t_nw10"] >= h)
    json.dump(summary, open(OUTPUT / "summary.json", "w"), indent=1)
    for sample in SAMPLES:
        s = summary[sample]
        print(f"{sample}: {s['days']} days, net {s['net_bps']} bps/day (NW10 t {s['t_nw10']}), gross {s['gross_bps']}, "
              f"turnover {s['turnover']}, Sharpe {s['sharpe']}, max drawdown {s['max_drawdown_pct']}%")
    print("holdout test:", summary["holdout_test"])
    return summary
