"""Demonstrate cached remote analytics with pymfs."""

import os
from datetime import timedelta
from pathlib import Path

from pymfs import FeatureStore


DEMO_DIRECTORY = Path(__file__).resolve().parent

store = FeatureStore(
    source=os.environ["HF_DESTINATION"],
    cache=DEMO_DIRECTORY / "feature_cache",
    token=os.environ["HF_TOKEN"],
    features=[
        "ohlcv:close",
        "ohlcv:volume",
        "sma:*",  # * will include all columns from "sma"
    ],
    # filters={"ticker": ["001", "017"]},
    start="2010-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    alignment="point_in_time"
    # alignment="exact", # no alignment, features are returned as-is, which is faster but may result in misaligned features
    # inmemory=True, # load all features into memory, which is faster but may use a lot of RAM
)
print(f"Remote feature store: {store.source}")
print(f"Local feature cache: {store.cache_path}")

symbols = store.query(
    """
    SELECT ticker, company_name, isin, market_code
    FROM symbology
    ORDER BY ticker
    LIMIT 5
    """
)
print("Symbology sample:")
print(symbols.fetchall())

result = store.query(
    """
    SELECT
        features.ticker,
        symbology.company_name,
        avg(features.close) AS average_close,
        sum(features.volume) AS total_volume
    FROM features
    LEFT JOIN symbology USING (ticker)
    GROUP BY features.ticker, symbology.company_name
    ORDER BY features.ticker
    """,
)
print("Feature summary with company information:")
print(result.df())

market_hours = store.table("markets").order("market_code, weekday")
print("Market hours:")
print(market_hours.df())

# Dont like SQL queries? Use the Python API to slice features and return a Pandas DataFrame.
# DuckDB supports pandas, polars, arrow etc. as output formats
frame = store.features(
    start="2024-01-02T08:00:00Z",
    end="2024-01-02T09:00:00Z",
    columns=["datetime", "ticker", "close", "volume", "sma50"],
    order_by=["datetime", "ticker"],
).to_df()
print(f"Pandas slice: {frame.shape}")
print(frame.head())

# Leaving columns and order params empty will grab all features and columns as is
frame = store.features(
    start="2024-01-02T08:00:00Z",
    end="2024-01-02T09:00:00Z",
).to_df()

# You can also leave out time params and just get everything.
# frame = store.features().to_df()
# But be careful with this on large datasets!
# Inside features() is a JOIN operator blocking operator that will load all features into memory.
# This is fine for small datasets, but for large datasets it can be slow and memory intensive.

# A better way it to use the feature_batches() method to stream data in smaller batches.
# Time windows bound the data DuckDB aligns for each training batch.
for batch in store.feature_batches(window=timedelta(days=1000)):
    df = batch.df()
    print(df.head())

# Or if you like to be more specific in time period and features (just like the features() method)
for batch in store.feature_batches(
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    columns=["datetime", "ticker", "close", "volume", "sma50"],
    sort_by=["datetime", "ticker"],
    window=timedelta(days=30)
):
    df = batch.df()
    print(df.head())
