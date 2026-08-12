# Sample Data Generation

This directory contains tools that generate sample data. The tools generate
deterministic Nasdaq Stockholm minute data. The tools calculate simple moving averages.
The tools build feature-store metadata. The tools upload the complete store to Hugging
Face.

The generated values are not real trades or prices. Do not use these values for live
trading or investment decisions.

## End-To-End Run

Set `HF_TOKEN` and `HF_DESTINATION` in the ignored `.env` file. Then, go to this
directory and run the pipeline:

```bash
cd demo
make data
```

The `data` target runs the `generate` and `upload` targets. They do these steps:

1. Generates yearly OHLCV Parquet partitions.
2. Generates SMA10, SMA20, SMA50, and SMA200 partitions.
3. Builds catalog version 3, dataset metadata, and the Hugging Face dataset card.
4. Creates or updates the configured private Hugging Face bucket or dataset.

By default, generation does not replace existing yearly files. It generates the same
data for the same date range, ticker range, and seed.

## Configuration

The Makefile passes the ignored `.env` file to commands that need Hugging Face
configuration. Set these values there:

| Setting | Purpose |
| --- | --- |
| `HF_DESTINATION` | Target `hf://buckets/OWNER/NAME` or `hf://datasets/OWNER/NAME` URI. |
| `HF_TOKEN` | Hugging Face token with write access. Never commit this value. |

Generation and run controls are Make variables with defaults:

| Setting | Purpose |
| --- | --- |
| `DATA_DIRECTORY` | Generated store directory, relative to this directory. |
| `START_DATE` | Inclusive ISO 8601 coverage start date. |
| `END_DATE` | Exclusive ISO 8601 coverage end date. |
| `TICKER_START` | First numeric ticker identifier. |
| `TICKER_COUNT` | Number of sequential tickers to generate. |
| `SEED` | Seed used for deterministic generation. |
| `OVERWRITE` | Controls replacement of existing yearly partitions. |
| `DRY_RUN` | Prevents the Hugging Face upload when set to `true`. |

Override settings on the command line. For example, this generates ten tickers and
previews the upload without writing to the remote store:

```bash
make data TICKER_COUNT=10 DRY_RUN=true
```

Use `make generate OVERWRITE=true` to replace existing partitions. Use `make upload`
to rebuild metadata and upload an already generated store.

## Individual Commands

Use this command to generate a small local test store. The store contains UTC OHLCV and
SMA time series. It also contains a keyed symbology table and a markets table without a
key.

```bash
uv run --group generation python gendata.py \
	--output data/ohlcv \
	--start 2024-01-01 \
	--end 2025-01-01 \
	--ticker-count 10 \
	--generate-sma \
	--sma-output data/sma \
	--symbology-output data/symbols/data.parquet \
	--markets-output data/markets/data.parquet
```

Use `--overwrite` to replace partitions that an older version generated. Catalog
version 3 accepts only Parquet time columns with the Arrow timezone `UTC`.

Use this command to build catalog metadata and read Parquet partition statistics:

```bash
uv run python build_catalog.py --data data \
	--store-name OWNER/NAME \
	--source hf://buckets/OWNER/NAME
```

This command writes catalog version 3 to `catalog.json`. It writes `metadata.json` for
each dataset. It also writes the generated store `README.md` in the configured data
directory. The `DATASETS` definition in `build_catalog.py` is the source for dataset and
feature semantics. Time-series datasets declare `time_column` and `series_keys`. Table
datasets can declare a composite `primary_key`. They can also omit `primary_key`.

Use this command to examine an upload without a remote write:

```bash
uv run --env-file .env python hfupload.py \
	--data data \
	--destination hf://buckets/OWNER/NAME \
	--dry-run
```

The uploader supports Hugging Face buckets and dataset repositories. The uploader
rebuilds the catalog and the generated dataset card before it transfers files.

## Library Demo

Run the remote cache and query example from this directory:

```bash
uv run --env-file .env --extra pandas python demo.py
```

The example reads `HF_DESTINATION` and `HF_TOKEN` from the environment. It stores
downloaded fragments in `demo/feature_cache`. The launch directory does not change
the cache location.

## Testing

Run the generation tool tests from this directory:

```bash
uv run --group test --group generation python -m pytest -q test_generate_data.py
```
