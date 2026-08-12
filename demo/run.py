"""Generate the complete feature store and upload it to Hugging Face."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from gendata import (
    generate_dataset,
    generate_markets,
    generate_sma_dataset,
    generate_symbology,
    ticker_values,
    write_table_dataset,
)
from hfupload import upload_data


def required(environment: Mapping[str, str], name: str) -> str:
    """Return a required, non-empty configuration value."""
    value = environment.get(name, "")
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def boolean(environment: Mapping[str, str], name: str) -> bool:
    """Parse a required boolean configuration value."""
    value = required(environment, name).lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


def main() -> None:
    environment = os.environ
    demo_directory = Path(__file__).resolve().parent
    data_directory = (demo_directory / required(environment, "DATA_DIRECTORY")).resolve()
    ohlcv_directory = data_directory / "ohlcv"
    sma_directory = data_directory / "sma"
    overwrite = boolean(environment, "OVERWRITE")
    ticker_start = int(required(environment, "TICKER_START"))
    ticker_count = int(required(environment, "TICKER_COUNT"))

    generate_dataset(
        output=ohlcv_directory,
        start=date.fromisoformat(required(environment, "START_DATE")),
        end=date.fromisoformat(required(environment, "END_DATE")),
        ticker_start=ticker_start,
        ticker_count=ticker_count,
        seed=int(required(environment, "SEED")),
        overwrite=overwrite,
    )
    generate_sma_dataset(ohlcv_directory, sma_directory, overwrite)
    write_table_dataset(
        data_directory / "symbols" / "data.parquet",
        generate_symbology(ticker_values(ticker_start, ticker_count)),
        overwrite,
    )
    write_table_dataset(
        data_directory / "markets" / "data.parquet",
        generate_markets(),
        overwrite,
    )
    upload_data(
        data_directory=data_directory,
        destination=required(environment, "HF_DESTINATION"),
        token=required(environment, "HF_TOKEN"),
        dry_run=boolean(environment, "DRY_RUN"),
    )


if __name__ == "__main__":
    main()