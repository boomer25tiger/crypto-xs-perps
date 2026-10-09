"""Monthly walk-forward of the two forecasting models.

Each month is scored by models trained on every panel row at least 7 days before the month starts, so no
training target overlaps the test month. The last 60 training days choose the ridge penalty and the number of
boosting rounds, by the top-minus-bottom decile spread of next-day spot returns; the chosen setting is then
refit on the whole training window.

The target is a horizon-weighted next-week return. Each day's demeaned next-day returns are divided by that
day's cross-sectional standard deviation, days 1 to 5 ahead are weighted 0.5, 0.25, 0.125, 0.0625 and 0.03125,
and the result is clipped at +/-5.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .config import FIRST_TEST_MONTH, LAST_TEST_MONTH, PANEL, SCORES
from .features import FEATURES, SQUEEZE, gauss_rank

ALPHAS = [0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
TREES = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=1.0, random_state=0,
             early_stopping=False)
ROUNDS = range(25, 401, 25)
HORIZON = [0.5, 0.25, 0.125, 0.0625, 0.03125]
EMBARGO_DAYS, VALIDATION_DAYS, CLIP = 7, 60, 5.0


def read_blocks(kind):
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PANEL / f"*_{kind}.parquet")))]).sort_index()
    return D[~D.index.duplicated()]


def horizon_target(next_ret):
    demeaned = (next_ret - next_ret.groupby(level="day").transform("mean")).unstack()
    W = demeaned.reindex(pd.date_range(demeaned.index.min(), demeaned.index.max(), freq="D"))
    W = W.div(next_ret.groupby(level="day").std().reindex(W.index), axis=0)
    num, den = 0, 0
    for k, w in enumerate(HORIZON):
        sh = W.shift(-k)
        num = num + (w * sh).fillna(0)
        den = den + w * sh.notna()
    return (num / den.where(den > 0)).where(W.notna()).stack().reindex(next_ret.index).values


def load_panel():
    """Feature matrix (27 daily features and 5 ranked squeeze features), target and next-day spot return."""
    Z = read_blocks("daily")
    H = read_blocks("squeeze").reindex(Z.index)
    X = np.column_stack([Z[FEATURES].values, pd.DataFrame({k: gauss_rank(H[k]) for k in SQUEEZE}, index=Z.index).fillna(0.0).values])
    return Z.index, X, horizon_target(Z.next_ret), Z.next_ret.values


def decile_spread(pred, ret, day, min_n=30):
    df = pd.DataFrame({"s": pred, "r": ret, "d": day})
    q = df.groupby("d").s.rank(pct=True)
    n = df.groupby("d").s.transform("size")
    hi = df.r.where((q > 0.9) & (n >= min_n)).groupby(df.d).mean()
    lo = df.r.where((q <= 0.1) & (n >= min_n)).groupby(df.d).mean()
    return float((hi - lo).mean())


class ColumnSubset:
    """A model fit on the columns that vary in its training rows; constant columns carry zero weight."""
    def __init__(self, keep, model):
        self.keep, self.model = keep, model

    def predict(self, X):
        return self.model.predict(np.asarray(X)[:, self.keep])


def fit_month(X, y, ret, day, train, fit, val):
    def target(m):
        return np.clip(y[m], -CLIP, CLIP)

    def score(p, m):
        v = decile_spread(p, ret[m], day[m])
        return -np.inf if not np.isfinite(v) else float(v)

    kf, kt = X[fit].std(0) > 0, X[train].std(0) > 0
    best = max((score(Ridge(alpha=a).fit(X[fit][:, kf], target(fit)).predict(X[val][:, kf]), val), a) for a in ALPHAS)
    ridge = ColumnSubset(kt, Ridge(alpha=best[1]).fit(X[train][:, kt], target(train)))
    info = {"alpha": best[1], "ridge_val": best[0]}
    gb, sc = HistGradientBoostingRegressor(max_iter=ROUNDS[0], warm_start=True, **TREES), []
    for it in ROUNDS:
        gb.set_params(max_iter=it).fit(X[fit], target(fit))
        sc.append((score(gb.predict(X[val]), val), it))
    best = max(sc)
    trees = HistGradientBoostingRegressor(max_iter=best[1], **TREES).fit(X[train], target(train))
    info.update(rounds=best[1], trees_val=best[0])
    return ridge, trees, info


def walk_forward(months=None):
    """Scores every month from 2021-01 to 2026-08; each finished month is saved, so a rerun resumes."""
    index, X, y, ret = load_panel()
    day = index.get_level_values("day")
    os.makedirs(SCORES, exist_ok=True)
    for m0 in months if months is not None else pd.date_range(FIRST_TEST_MONTH, LAST_TEST_MONTH, freq="MS"):
        f = SCORES / f"{m0:%Y-%m}.parquet"
        if f.exists():
            continue
        test = np.asarray((day >= m0) & (day < m0 + pd.offsets.MonthBegin(1)))
        train = np.asarray(day <= m0 - pd.Timedelta(days=EMBARGO_DAYS))
        cut = day[train].max() - pd.Timedelta(days=VALIDATION_DAYS)
        fit, val = train & np.asarray(day <= cut), train & np.asarray(day > cut)
        ridge, trees, info = fit_month(X, y, ret, day, train, fit, val)
        out = pd.DataFrame({"ridge": ridge.predict(X[test]), "trees": trees.predict(X[test])}, index=index[test])
        out.to_parquet(f)
        print(f"{m0:%Y-%m}: alpha {info['alpha']:g}, {info['rounds']} rounds, {int(train.sum())} training rows", flush=True)


def load_scores():
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SCORES / "*.parquet")))]).sort_index()
