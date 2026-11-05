# regime-detection

Regime-conditional VaR/CVaR pipeline for energy markets (WTI, Brent, Henry Hub) using macro indicators (10Y-2Y yield spread, DXY, VIX).

## Layout

```
data/
  fomc_meetings.csv          FOMC meeting dates
  opec_meetings.csv          OPEC meeting dates
notebooks/
  demo.ipynb                 End-to-end analysis (orchestration only — all logic lives in modules)
scripts/
  build_meeting_dates.py     Scrapes FOMC/OPEC calendars into data/
src/regime_detection/
  ingestion.py               EIA + FRED API pulls, merged into one weekly panel
  features.py                Log returns, diffs, lags, rolling volatility, cross-commodity spreads
  stationarity.py            ADF / KPSS joint classification and transform pipeline
  regression.py              OLS + HAC, Rolling OLS, Ljung-Box, Durbin-Watson diagnostics
  changepoints.py            PELT changepoint detection with CROPS penalty search, regime labeling
  risk.py                    Rolling VaR, regime-conditional VaR/CVaR, breach rates
  backtest.py                Kupiec proportion-of-failures LR test
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
from regime_detection.regression import fit_commodity_models, fit_rolling_commodity_models
from regime_detection.changepoints import run_crops, select_plateau_changepoints, label_regimes
from regime_detection.risk import add_rolling_var_columns, regime_conditional_risk
from regime_detection.backtest import kupiec_regime_conditional, kupiec_unconditional
```

Or run `notebooks/demo.ipynb` for the full pipeline.
