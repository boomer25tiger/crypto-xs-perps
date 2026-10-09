"""Run the study stage by stage.

    python pipeline.py download    Binance hourly klines and funding, plus S&P 500 closes
    python pipeline.py universe    monthly top-100 universe
    python pipeline.py features    features and perp returns, block by block
    python pipeline.py fit         monthly walk-forward of ridge and boosted trees, 2021-01 to 2026-08
    python pipeline.py report      design and holdout books, summary.json
    python pipeline.py figures     README figures

Each stage skips work already on disk; delete a file under output/ to redo it.
"""
import sys

from crypto_xs import config


def features():
    from crypto_xs import features as ft, perps
    for name, d0, d1 in config.chunks():
        if not (config.PANEL / f"{name}_daily.parquet").exists():
            ft.daily_panel(name, d0, d1)
        if not (config.PANEL / f"{name}_squeeze.parquet").exists():
            ft.squeeze_features(name, d0, d1)
        if not (config.PANEL / f"{name}_perps.parquet").exists():
            perps.perp_returns(name, d0, d1)


def main(stage):
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    if stage == "download":
        from crypto_xs import download
        download.run()
    elif stage == "universe":
        from crypto_xs import universe
        universe.build()
    elif stage == "features":
        features()
    elif stage == "fit":
        from crypto_xs import models
        models.walk_forward()
    elif stage == "report":
        from crypto_xs import report
        report.run()
    elif stage == "figures":
        from crypto_xs import plots
        plots.run()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
