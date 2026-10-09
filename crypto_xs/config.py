"""Paths, dates and constants shared by every stage."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("XS_DATA", ROOT / "data"))
OUTPUT = Path(os.environ.get("XS_OUTPUT", ROOT / "output"))
KLINES = DATA / "klines"
PANEL = OUTPUT / "panel"
SCORES = OUTPUT / "scores"


def utc(s):
    return pd.Timestamp(s, tz="UTC")


PANEL_START = utc("2020-05-01")    # first decision day in the model panel
DATA_END = utc("2026-09-01")       # hourly bars are used up to here
FIRST_TEST_MONTH = utc("2021-01-01")
LAST_TEST_MONTH = utc("2026-08-01")

# Decision days of each sample. A decision day d holds its book over day d+1. The holdout starts on
# 2022-07-01, the date it was sealed from until its single read. The design book's last holding day is
# 2022-06-29: the next one closes with the 2022-07-01 00:00 funding settlement, which sat behind the seal.
SAMPLES = {"design": (utc("2021-01-01"), utc("2022-06-28")),
           "holdout": (utc("2022-07-01"), utc("2026-08-30"))}

# Cost per unit of turnover. The spread starts from an 11.0 bp quoted half-spread for Binance coins trading about
# $20M a day (Wu, Foley and Svec 2024), scaled by 1.14 for perpetuals over spot and 1.031 for the funding hour
# (Ruan and Streltsov), so 12.93 bp one way. The taker fee adds 4 bp in the design period and 5 bp in the holdout.
COST = {"design": 16.93e-4, "holdout": 17.93e-4}

# Multiple-testing hurdle for the holdout t statistic. The development record holds 142 trials with mean
# pairwise correlation 0.71; the hurdle is the larger of 2.0 and sqrt(1 - rho) times the expected maximum of
# N standard normals.
TRIALS, TRIAL_CORRELATION = 142, 0.71

# Bases left out of the universe (stablecoins and fiat, leveraged tokens, wrapped or staked coins).
STABLE = {"AUD", "BFUSD", "BUSD", "DAI", "EUR", "FDUSD", "FRAX", "GBP", "PAX", "PAXG", "SUSD", "TUSD", "USD1",
          "USDC", "USDE", "USDP", "USDS", "USDSB", "USDSOLD", "UST", "USTC", "XUSD", "AEUR", "EURI", "RLUSD"}
LEVERAGED = {f"{c}{s}" for c in ["1INCH", "AAVE", "ADA", "BCH", "BNB", "BTC", "DOT", "EOS", "ETH", "FIL", "LINK",
                                 "LTC", "SUSHI", "SXP", "TRX", "UNI", "XLM", "XRP", "XTZ", "YFI"]
             for s in ["UP", "DOWN"]} | {"BNBBULL", "BNBBEAR", "EOSBULL", "EOSBEAR", "ETHBULL", "ETHBEAR",
                                         "XRPBULL", "XRPBEAR"}
WRAPPED = {"BETH", "WBETH", "WBTC"}
# BULLUSDT and BEARUSDT (leveraged BTC tokens) stay in the volume ranking that sets the universe and are
# removed from the traded panel.
NOT_TRADED = {"BULLUSDT", "BEARUSDT"}


def eligible(symbol):
    if not symbol.endswith("USDT"):
        return False
    base = symbol[:-4]
    return base not in STABLE and base not in LEVERAGED and base not in WRAPPED


def chunks():
    """Half-year blocks of decision days. Features are built block by block with enough history loaded
    before each block that every rolling window is complete, so the blocks join without seams."""
    edges = [PANEL_START] + list(pd.date_range(utc("2021-01-01"), DATA_END, freq="6MS")) + [DATA_END]
    return [(a.strftime("%Y-%m"), a, b - pd.Timedelta(days=1)) for a, b in zip(edges[:-1], edges[1:])]
