# Data acquisition and provenance

Raw market prices are not distributed in this repository. This directory contains documentation only until you run the project locally.

## Source and request

The script obtains daily Yahoo Finance prices using `yfinance` with `auto_adjust=True`. It reads the adjusted `Close` field for this fixed teaching universe:

```text
AAPL, MSFT, AMZN, GOOGL, JPM
start: 2019-01-01
end:   2026-01-01, exclusive
```

The original local research snapshot had 1,760 complete trading dates from 2019-01-02 through 2025-12-31. The checked-in reports record results from that snapshot; the price snapshot itself is excluded.

## Download and cache locally

After installing the core requirements, run from the repository root:

```bash
python equity_lfm_project.py
```

When a matching local cache is absent, the script downloads data and writes:

```text
data/yahoo_adjusted_close.csv
data/yahoo_download_metadata.json
```

The metadata records source, request dates, download time, and the observed data range. A compatible cache is reused on later runs. To request fresh prices:

```bash
python equity_lfm_project.py --refresh-data
```

The script validates complete positive prices before saving a new snapshot. Missing or unusable market data raise errors. It does not replace failed downloads with simulated observations. Synthetic data are used only by the offline mathematical self-tests.

## Reproducibility limits

This repository provides a reproducible workflow and recorded configuration, not an exact public archive of the vendor inputs. Yahoo can revise adjusted history, and download availability can change. A fresh run may therefore differ from the checked-in reports even with identical code and parameters.

The five stocks were selected as a fixed small example and all survive today. This is not a point-in-time historical universe and does not remove survivorship or selection bias. Adjusted historical closes are not a strict point-in-time information archive or a guarantee of executable closing prices.

The two structural risk groups are static analyst-defined proxies. They are not historical vendor industry classifications. See [methodology](../docs/METHODOLOGY.md) for the full model and execution assumptions.

Local cache files are ignored by Git. Keep raw vendor data out of a manual browser upload as well, because browser uploads do not apply `.gitignore`. The public [run configuration](../reports/run_config.json) describes the recorded experiment without distributing the underlying prices.
