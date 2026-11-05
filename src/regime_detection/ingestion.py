"""Data ingestion: EIA, FRED, and meeting-date merging."""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# EIA helpers
# ---------------------------------------------------------------------------

_EIA_ENDPOINTS = {
    "petroleum_spot": "petroleum/pri/spt",
    "natural_gas_futures": "natural-gas/pri/fut",
    "petroleum_stock": "petroleum/stoc/wstk",
}


def fetch_eia_series(
    series_ids: list[str],
    endpoint: str = "petroleum_spot",
    frequency: str = "weekly",
    length: int = 5000,
) -> pd.DataFrame:
    """Fetch one or more EIA series and return a wide DataFrame indexed by period.

    Parameters
    ----------
    series_ids : list[str]
        EIA series identifiers, e.g. ``["RBRTE", "RWTC"]``.
    endpoint : str
        Key into ``_EIA_ENDPOINTS`` (petroleum_spot | natural_gas_futures | petroleum_stock).
    frequency : str
        "daily", "weekly", or "monthly".
    length : int
        Maximum number of records to return (EIA caps at 5000).
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise EnvironmentError("EIA_API_KEY not set in environment or .env")

    path = _EIA_ENDPOINTS[endpoint]
    facets = "".join(f"&facets[series][]={sid}" for sid in series_ids)
    url = (
        f"https://api.eia.gov/v2/{path}/data/"
        f"?api_key={api_key}"
        f"&frequency={frequency}"
        f"&data[0]=value"
        f"{facets}"
        f"&sort[0][column]=period&sort[0][direction]=desc"
        f"&length={length}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data["response"]["data"])
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["period", "value", "series"]]

    df_wide = df.pivot(index="period", columns="series", values="value").dropna()
    return df_wide.sort_index()


# Convenience wrappers (preserve the old call signatures)

def fetch_eia_series_pet(series_ids=None, frequency="weekly", length=5000):
    return fetch_eia_series(
        series_ids or ["RBRTE", "RWTC"],
        endpoint="petroleum_spot",
        frequency=frequency,
        length=length,
    )


def fetch_eia_series_gas(series_ids=None, frequency="weekly", length=5000):
    return fetch_eia_series(
        series_ids or ["RNGWHHD"],
        endpoint="natural_gas_futures",
        frequency=frequency,
        length=length,
    )


def fetch_eia_stock(series_ids=None, frequency="weekly", length=5000):
    return fetch_eia_series(
        series_ids or ["WCESTUS1"],
        endpoint="petroleum_stock",
        frequency=frequency,
        length=length,
    )


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_ids: list[str] | None = None,
    frequency: str = "W-FRI",
) -> pd.DataFrame:
    """Fetch series from FRED and resample to *frequency*."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set in environment or .env")

    series_ids = series_ids or ["T10Y2Y", "VIXCLS", "DTWEXBGS"]
    fred = Fred(api_key=api_key)

    series_dict = {sid: fred.get_series(series_id=sid) for sid in series_ids}

    df = pd.DataFrame(series_dict)
    df.index = pd.to_datetime(df.index)
    df.index.name = "period"
    df = df.resample(rule=frequency).last().dropna()
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def get_merged_df(
    eia_spot_pet_df=None,
    eia_spot_gas_df=None,
    eia_stock_df=None,
    fred_df=None,
) -> pd.DataFrame:
    """Pull all sources and merge into a single weekly panel."""
    if eia_spot_pet_df is None:
        eia_spot_pet_df = fetch_eia_series_pet()
    if eia_spot_gas_df is None:
        eia_spot_gas_df = fetch_eia_series_gas()
    if eia_stock_df is None:
        eia_stock_df = fetch_eia_stock()
    if fred_df is None:
        fred_df = fetch_fred_series()

    df = (
        eia_spot_pet_df
        .merge(fred_df, left_index=True, right_index=True, how="inner")
        .merge(eia_stock_df, left_index=True, right_index=True, how="inner")
        .merge(eia_spot_gas_df, left_index=True, right_index=True, how="inner")
        .reset_index()
        .sort_values("period")
    )

    # OPEC meeting dates
    opec = pd.read_csv(DATA_DIR / "opec_meetings.csv", parse_dates=["date"]).sort_values("date")
    df = pd.merge_asof(
        df, opec.assign(last_opec=opec["date"]),
        left_on="period", right_on="date", direction="backward",
    )
    df["days_since_opec"] = (df["period"] - df["last_opec"]).dt.days

    # FOMC meeting dates
    fomc = pd.read_csv(DATA_DIR / "fomc_meetings.csv", parse_dates=["date"]).sort_values("date")
    df = pd.merge_asof(
        df, fomc.assign(last_fomc=fomc["date"]),
        left_on="period", right_on="date", direction="backward",
    )
    df["days_since_fomc"] = (df["period"] - df["last_fomc"]).dt.days

    df = df.drop(columns=["date_x", "date_y", "last_fomc", "last_opec"])
    return df.set_index("period")
