"""Demonstrate cached remote analytics with pymfs."""

import os
from pathlib import Path

from pymfs import FeatureStore
from run import required


DEMO_DIRECTORY = Path(__file__).resolve().parent

fs = FeatureStore(
    source=required(os.environ, "HF_DESTINATION"),
    cache=DEMO_DIRECTORY / "feature_cache",
    token=required(os.environ, "HF_TOKEN"),
    features=[
        "ohlcv:close",
        "ohlcv:volume",
        "sma:sma10",
        "sma:sma20",
        "sma:sma50",
        "sma:sma200",
    ],
    #filters={"ticker": ["001", "017"]},
    start="2010-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    alignment="point_in_time", # slow for larger datasets, but ensures that features are aligned to the same timestamp
    #alignment="exact", # no alignment, features are returned as-is, which is faster but may result in misaligned features
    #inmemory=True, # load all features into memory, which is faster but may use a lot of RAM
)
print(f"Remote feature store: {fs.source}")
print(f"Local feature cache: {fs.cache_path}")

symbols = fs.query(
    """
    SELECT ticker, company_name, isin, market_code
    FROM symbology
    ORDER BY ticker
    LIMIT 5
    """
)
print("Symbology sample:")
print(symbols.fetchall())

result = fs.query(
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

market_hours = fs.query(
    """
    SELECT market_code, weekday, opens_at, closes_at, timezone
    FROM markets
    ORDER BY market_code, weekday
    """,
)
print("Market hours:")
print(market_hours.df())

frame = fs.query(
        """
        SELECT * FROM features
        WHERE datetime >= TIMESTAMPTZ '2024-01-02T08:00:00Z'
            AND datetime < TIMESTAMPTZ '2024-01-02T09:00:00Z'
        """
).df()
print(f"Pandas slice: {frame.shape}")
print(frame.head())
