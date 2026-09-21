# regime-detection

Regime Detection + VaR and CVaR using WTI, Brent (Portfolio), 10Y - 2Y yield spreads, DXY, and VIX (To inform the model).

## Layout

```
data/        CSVs — FOMC/OPEC meeting dates and the cleaned merged panel
notebooks/   data_ingestion.ipynb (pulling + cleaning), demo.ipynb (analysis)
scripts/     build_meeting_dates.py — scrapes the FOMC/OPEC calendars into data/
src/regime_detection/
  ingestion.py     EIA + FRED pulls, merged into one weekly panel
  features.py      log returns, diffs, lags, rolling vol
  stationarity.py  ADF / KPSS tests and the diagnostic pipeline
```

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
```

Put your API keys in a `.env` at the repo root:

```
EIA_API_KEY=...
FRED_API_KEY=...
```

## Usage

```python
from regime_detection.ingestion import get_merged_df
from regime_detection.features import feature_pipeline
from regime_detection.stationarity import stationarity_pipeline
```
