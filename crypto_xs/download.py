"""Download the public inputs. Binance hourly klines for spot and USDT-M perpetuals and every USDT-M funding
settlement come from the archive at data.binance.vision, one zip per symbol and month, each checked against its
published SHA-256. The S&P 500 daily close comes from Yahoo Finance. No API key is needed.
"""
import csv
import hashlib
import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from .config import DATA, KLINES, eligible

ARCHIVE = "https://data.binance.vision/"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_PATH = {"spot": "data/spot/monthly/klines/", "um": "data/futures/um/monthly/klines/"}
FUNDING_PATH = "data/futures/um/monthly/fundingRate/"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
           "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]
FLOATS = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume",
          "taker_buy_quote_volume"]
LAST_MONTH = "2026-08"

session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16))


def get(url, params=None, tries=4):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.0 * (k + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def listing(prefix, delimiter=None):
    """Keys (or sub-directories, with a delimiter) under a prefix of the archive bucket."""
    out, marker = [], ""
    while True:
        p = {"prefix": prefix}
        if delimiter:
            p["delimiter"] = delimiter
        if marker:
            p["marker"] = marker
        body = get(LISTING, p).decode()
        found = re.findall(r"<Prefix>([^<]+)</Prefix>", body) if delimiter else re.findall(r"<Key>([^<]+)</Key>", body)
        out += [x for x in found if x != prefix]
        if "<IsTruncated>true</IsTruncated>" not in body:
            return out
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", body)
        marker = nxt.group(1) if nxt else out[-1]


def symbols(market):
    return sorted(d.rstrip("/").split("/")[-1] for d in listing(KLINE_PATH[market], "/"))


def months(market, symbol):
    keys = listing(f"{KLINE_PATH[market]}{symbol}/1h/")
    found = (re.search(r"-1h-(\d{4}-\d{2})\.zip$", k) for k in keys)
    return sorted(m.group(1) for m in found if m and m.group(1) <= LAST_MONTH)


def verified_zip(key):
    blob = get(ARCHIVE + key)
    if blob is None:
        return None
    want = get(ARCHIVE + key + ".CHECKSUM").decode().split()[0].lower()
    if hashlib.sha256(blob).hexdigest() != want:
        raise RuntimeError(f"checksum mismatch: {key}")
    return blob


def parse_klines(blob):
    """Spot files have no header and switch to microsecond timestamps in 2025; perp files have a header."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    raw = z.read(z.namelist()[0]).decode()
    header = not raw.split("\n", 1)[0].split(",")[0].strip('"').replace(".", "").isdigit()
    df = pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS, skiprows=1 if header else 0)
    for c in ["open_time", "close_time"]:
        t = df[c].astype("int64")
        df[c] = t // 1000 if len(str(int(t.iloc[0]))) == 16 else t
    for c in FLOATS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype("int64")
    return df.drop(columns="ignore")


def download_symbol(market, symbol, pool):
    ms = months(market, symbol)
    blobs = pool.map(lambda m: verified_zip(f"{KLINE_PATH[market]}{symbol}/1h/{symbol}-1h-{m}.zip"), ms)
    frames = [parse_klines(b) for b in blobs if b is not None]
    if frames:
        df = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
        df.to_parquet(KLINES / market / f"{symbol}.parquet", index=False)
    return len(frames)


def klines():
    spot = [s for s in symbols("spot") if eligible(s)]
    perps = sorted(set(symbols("um")) & set(spot))
    with ThreadPoolExecutor(16) as pool:
        for market, names in [("spot", spot), ("um", perps)]:
            (KLINES / market).mkdir(parents=True, exist_ok=True)
            for i, s in enumerate(names, 1):
                if not (KLINES / market / f"{s}.parquet").exists():
                    download_symbol(market, s, pool)
                if i % 50 == 0:
                    print(f"{market}: {i}/{len(names)} symbols", flush=True)


def funding():
    """Every settlement for the perps on disk, with its interval (Binance moved some contracts from 8-hour
    to 4-hour or 1-hour settlement from late 2023)."""
    perps = sorted(f.stem for f in (KLINES / "um").glob("*.parquet"))
    span = {s: pd.to_datetime(pd.read_parquet(KLINES / "um" / f"{s}.parquet", columns=["open_time"]).open_time,
                              unit="ms").dt.strftime("%Y-%m").agg(["min", "max"]).tolist() for s in perps}
    jobs = [(s, str(m)) for s in perps for m in pd.period_range(span[s][0], min(span[s][1], LAST_MONTH), freq="M")]

    def one(job):
        s, m = job
        blob = get(f"{ARCHIVE}{FUNDING_PATH}{s}/{s}-fundingRate-{m}.zip")
        if blob is None:
            return []
        rows = []
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                for row in csv.reader(line.decode() for line in z.open(name)):
                    if row and row[0].strip().isdigit():
                        rows.append([s, row[0].strip(), row[1].strip() if len(row) >= 3 else "", row[-1].strip()])
        return rows

    with ThreadPoolExecutor(8) as pool, open(DATA / "funding.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "calc_time_ms", "funding_interval_hours", "funding_rate"])
        for rows in pool.map(one, jobs):
            w.writerows(rows)
    print("funding:", len(jobs), "symbol-months requested")


def sp500():
    import yfinance as yf
    px = yf.download("^GSPC", start="2017-01-01", end="2026-09-16", auto_adjust=False, progress=False)
    s = px["Adj Close"].squeeze()
    dates = s.index.tz_localize(None) if s.index.tz is not None else s.index
    pd.DataFrame({"date": dates, "adj_close": s.values}).to_parquet(DATA / "sp500.parquet", index=False)


def run():
    DATA.mkdir(parents=True, exist_ok=True)
    klines()
    funding()
    sp500()
