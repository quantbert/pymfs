"""Generate deterministic time-series and table datasets for a synthetic store.

The default invocation writes data for tickers ``000`` through ``099`` from
2010-01-01 (inclusive) to 2025-01-01 (exclusive). Output is partitioned by
UTC year so a partial run can be generated, inspected, or resumed. It also
writes keyed symbology and non-keyed market-hours tables.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


MARKET_TIMEZONE = ZoneInfo("Europe/Stockholm")
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(17, 30)
SMA_WINDOWS = (10, 20, 50, 200)
MARKET_CALENDAR = xcals.get_calendar("XSTO")
SCHEMA = pa.schema(
    [
        pa.field("datetime", pa.timestamp("us", tz="UTC")),
        pa.field("ticker", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
SMA_SCHEMA = pa.schema(
    [
        pa.field("datetime", pa.timestamp("us", tz="UTC")),
        pa.field("ticker", pa.string()),
        *(pa.field(f"sma{window}", pa.float64()) for window in SMA_WINDOWS),
    ]
)
SYMBOLOGY_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("isin", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("market_code", pa.string(), nullable=False),
    ]
)
MARKETS_SCHEMA = pa.schema(
    [
        pa.field("market_code", pa.string(), nullable=False),
        pa.field("weekday", pa.string(), nullable=False),
        pa.field("opens_at", pa.time32("s"), nullable=False),
        pa.field("closes_at", pa.time32("s"), nullable=False),
        pa.field("timezone", pa.string(), nullable=False),
    ]
)


def parse_date(value: str) -> date:
    """Parse an ISO-8601 date used for an inclusive/exclusive date range."""
    return date.fromisoformat(value)


def trading_days(start: date, end: date) -> list[date]:
    """Return Nasdaq Stockholm (XSTO) sessions in ``[start, end)``."""
    if end <= start:
        return []
    sessions = MARKET_CALENDAR.sessions_in_range(start, end - timedelta(days=1))
    return [session.date() for session in sessions]


def session_minutes(days: list[date]) -> list[datetime]:
    """Create UTC minute-bar timestamps for Stockholm sessions from 09:00 through 17:29."""
    minutes: list[datetime] = []
    for trading_day in days:
        timestamp = datetime.combine(trading_day, MARKET_OPEN, MARKET_TIMEZONE)
        close_timestamp = datetime.combine(trading_day, MARKET_CLOSE, MARKET_TIMEZONE)
        while timestamp < close_timestamp:
            minutes.append(timestamp.astimezone(UTC))
            timestamp += timedelta(minutes=1)
    return minutes


def ticker_values(ticker_start: int, ticker_count: int) -> list[str]:
    """Return zero-padded synthetic tickers after validating their range."""
    if not 0 <= ticker_start <= 999 or not 1 <= ticker_count <= 1_000 - ticker_start:
        raise ValueError("ticker range must stay within 000 through 999")
    return [f"{number:03d}" for number in range(ticker_start, ticker_start + ticker_count)]


def generate_symbology(tickers: list[str]) -> pa.Table:
    """Generate one deterministic symbology row for every ticker."""
    return pa.table(
        {
            "ticker": tickers,
            "isin": [f"SE{int(ticker):010d}" for ticker in tickers],
            "cik": [f"{int(ticker) + 1:010d}" for ticker in tickers],
            "company_name": [f"Example Company {ticker}" for ticker in tickers],
            "description": [
                f"Synthetic company record for ticker {ticker}." for ticker in tickers
            ],
            "market_code": ["XSTO"] * len(tickers),
        },
        schema=SYMBOLOGY_SCHEMA,
    )


def generate_markets() -> pa.Table:
    """Generate market-hours rows without declaring or enforcing a primary key."""
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    rows = [
        ("XSTO", weekday, time(9, 0), time(17, 30), "Europe/Stockholm")
        for weekday in weekdays
    ]
    rows.extend(
        ("XNYS", weekday, time(9, 30), time(16, 0), "America/New_York")
        for weekday in weekdays
    )
    return pa.Table.from_pylist(
        [
            {
                "market_code": market_code,
                "weekday": weekday,
                "opens_at": opens_at,
                "closes_at": closes_at,
                "timezone": timezone,
            }
            for market_code, weekday, opens_at, closes_at, timezone in rows
        ],
        schema=MARKETS_SCHEMA,
    )


def write_table_dataset(path: Path, table: pa.Table, overwrite: bool) -> None:
    """Atomically write a single-file table dataset."""
    if path.exists() and not overwrite:
        print(f"Skipping existing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".parquet.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Wrote {table.num_rows:,} rows to {path}")


def generate_bars(ticker: str, year: int, timestamps: list[datetime], seed: int) -> pa.Table:
    """Generate deterministic, internally consistent OHLCV bars for one partition."""
    generator = np.random.default_rng(np.random.SeedSequence([seed, int(ticker), year]))
    row_count = len(timestamps)

    starting_price = generator.uniform(20.0, 500.0)
    close = starting_price * np.exp(np.cumsum(generator.normal(0.0, 0.0015, row_count)))
    open_price = np.empty(row_count)
    open_price[0] = starting_price
    open_price[1:] = close[:-1] * np.exp(generator.normal(0.0, 0.0003, row_count - 1))

    upper_spread = generator.uniform(0.0, 0.002, row_count)
    lower_spread = generator.uniform(0.0, 0.002, row_count)
    high = np.maximum(open_price, close) * (1.0 + upper_spread)
    low = np.minimum(open_price, close) * (1.0 - lower_spread)
    volume = generator.integers(100, 1_000_001, row_count, dtype=np.int64)

    return pa.table(
        {
            "datetime": pa.array(timestamps, type=SCHEMA.field("datetime").type),
            "ticker": pa.array([ticker] * row_count, type=pa.string()),
            "open": pa.array(open_price),
            "high": pa.array(high),
            "low": pa.array(low),
            "close": pa.array(close),
            "volume": pa.array(volume),
        },
        schema=SCHEMA,
    )


def years_in_range(start: date, end: date) -> list[tuple[int, date, date]]:
    """Split an inclusive/exclusive date range into calendar-year ranges."""
    ranges: list[tuple[int, date, date]] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, date(year, 1, 1))
        year_end = min(end, date(year + 1, 1, 1))
        if year_start < year_end:
            ranges.append((year, year_start, year_end))
    return ranges


def generate_dataset(
    output: Path,
    start: date,
    end: date,
    ticker_start: int,
    ticker_count: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Write one yearly file, streaming one ticker at a time to bound memory use."""
    if end <= start:
        raise ValueError("end must be later than start")
    tickers = ticker_values(ticker_start, ticker_count)

    for year, year_start, year_end in years_in_range(start, end):
        timestamps = session_minutes(trading_days(year_start, year_end))
        if not timestamps:
            print(f"Skipping {year}: no XSTO trading sessions in requested range")
            continue
        destination = output / f"year={year}" / "data.parquet"
        if destination.exists() and not overwrite:
            print(f"Skipping existing {destination}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination = destination.with_suffix(".parquet.tmp")
        temporary_destination.unlink(missing_ok=True)
        try:
            with pq.ParquetWriter(temporary_destination, SCHEMA, compression="zstd") as writer:
                for ticker in tickers:
                    bars = generate_bars(ticker, year, timestamps, seed)
                    writer.write_table(bars, row_group_size=250_000)
            temporary_destination.replace(destination)
        except BaseException:
            temporary_destination.unlink(missing_ok=True)
            raise
        print(
            f"Wrote {len(timestamps) * ticker_count:,} rows for {ticker_count:,} tickers "
            f"to {destination}"
        )


def simple_moving_averages(close: np.ndarray, history: np.ndarray) -> dict[int, np.ndarray]:
    """Calculate SMAs, retaining only the preceding values needed by each window."""
    values = np.concatenate((history, close))
    history_length = len(history)
    averages: dict[int, np.ndarray] = {}
    for window in SMA_WINDOWS:
        result = np.full(len(close), np.nan)
        if len(values) >= window:
            cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
            all_averages = (cumulative[window:] - cumulative[:-window]) / window
            first_output = max(0, window - 1 - history_length)
            result[first_output:] = all_averages[-len(result) + first_output :]
        averages[window] = result
    return averages


def generate_sma_dataset(ohlcv_root: Path, output: Path, overwrite: bool) -> None:
    """Generate SMA10, SMA20, SMA50, and SMA200 datasets from yearly OHLCV files."""
    close_history: dict[str, np.ndarray] = {}
    source_files = sorted(ohlcv_root.glob("year=*/data.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No yearly OHLCV files found under {ohlcv_root}")

    for source in source_files:
        year_directory = source.parent.name
        destination = output / year_directory / "data.parquet"
        write_output = overwrite or not destination.exists()
        writer: pq.ParquetWriter | None = None
        temporary_destination = destination.with_suffix(".parquet.tmp")
        rows_written = 0
        try:
            if write_output:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_destination.unlink(missing_ok=True)
                writer = pq.ParquetWriter(temporary_destination, SMA_SCHEMA, compression="zstd")
            for batch in pq.ParquetFile(source).iter_batches(batch_size=250_000):
                table = pa.Table.from_batches([batch])
                tickers = table["ticker"].to_numpy(zero_copy_only=False)
                close = table["close"].to_numpy(zero_copy_only=False)
                for ticker in np.unique(tickers):
                    ticker_mask = tickers == ticker
                    ticker_close = close[ticker_mask]
                    history = close_history.get(ticker, np.array([], dtype=np.float64))
                    if writer is not None:
                        averages = simple_moving_averages(ticker_close, history)
                        writer.write_table(
                            pa.table(
                                {
                                    "datetime": table["datetime"].filter(pa.array(ticker_mask)),
                                    "ticker": pa.array(
                                        [ticker] * len(ticker_close), type=pa.string()
                                    ),
                                    **{
                                        f"sma{window}": pa.array(averages[window])
                                        for window in SMA_WINDOWS
                                    },
                                },
                                schema=SMA_SCHEMA,
                            ),
                            row_group_size=250_000,
                        )
                    close_history[ticker] = np.concatenate((history, ticker_close))[-199:]
                    rows_written += len(ticker_close)
            if writer is not None:
                writer.close()
                writer = None
                temporary_destination.replace(destination)
                print(f"Wrote {rows_written:,} SMA rows to {destination}")
            else:
                print(f"Loaded history from existing {destination}")
        except BaseException:
            if writer is not None:
                writer.close()
            temporary_destination.unlink(missing_ok=True)
            raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ohlcv"))
    parser.add_argument("--start", type=parse_date, default=date(2010, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date(2025, 1, 1))
    parser.add_argument("--ticker-start", type=int, default=0)
    parser.add_argument(
        "--ticker-count",
        type=int,
        default=100,
        help="Number of sequential tickers to generate (default: 100, or 000 through 099).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--generate-sma",
        action="store_true",
        help="Generate data/sma from the yearly OHLCV Parquet files after generating OHLCV.",
    )
    parser.add_argument(
        "--sma-output",
        type=Path,
        default=Path("data/sma"),
        help="Directory for generated SMA feature files.",
    )
    parser.add_argument(
        "--symbology-output",
        type=Path,
        default=Path("data/symbols/data.parquet"),
        help="Path for the keyed symbology table.",
    )
    parser.add_argument(
        "--markets-output",
        type=Path,
        default=Path("data/markets/data.parquet"),
        help="Path for the non-keyed market-hours table.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    settings = arguments()
    tickers = ticker_values(settings.ticker_start, settings.ticker_count)
    generate_dataset(
        output=settings.output,
        start=settings.start,
        end=settings.end,
        ticker_start=settings.ticker_start,
        ticker_count=settings.ticker_count,
        seed=settings.seed,
        overwrite=settings.overwrite,
    )
    if settings.generate_sma:
        generate_sma_dataset(settings.output, settings.sma_output, settings.overwrite)
    write_table_dataset(
        settings.symbology_output,
        generate_symbology(tickers),
        settings.overwrite,
    )
    write_table_dataset(
        settings.markets_output,
        generate_markets(),
        settings.overwrite,
    )