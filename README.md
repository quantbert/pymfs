# pymfs

`pymfs` (Python Minimal Feature Store) is a small Python library for querying Parquet
time series and general tables with DuckDB. It supports feature selection, local
caching, cross-dataset joins, point-in-time alignment, streaming, and analytical SQL.
A store can be in a local directory, a Hugging Face dataset repository, or a Hugging
Face bucket.

The library is for offline read operations. It does not calculate features. It does not
provide an online database, ingestion control, or a registry service. Producers and
`FeatureStore` use the catalog and the Parquet layout as their interface.

`pymfs` removes the operational overhead from feature retrieval. It provides the
minimum needed to select features and reference data, prepare a training dataset, and
continue with your work.

There are no servers to host, configure, or maintain. There is no service deployment,
registry, or ingestion control plane. A store is a catalog plus Parquet files.

## Installation

```bash
pip install pymfs
```

```python
from pymfs import FeatureStore
```

Install an optional dependency when you need pandas or Hugging Face Datasets output:

```bash
pip install "pymfs[pandas]"
pip install "pymfs[datasets]"
pip install "pymfs[all]"
```

Use Python 3.14 or newer.

## Core Model

A store contains one or more **datasets**. A `timeseries` dataset declares its event-time
column. It also declares zero or more series keys. A `table` dataset can declare a
composite primary key. It can also have no key.

A `FeatureStore` instance has a fixed data selection. Set the source, cache, features,
time coverage, filters, alignment, and in-memory mode during construction. You cannot
change these values after construction. Create a different `FeatureStore` for a
different feature set, interval, or filter scope. Multiple instances can use the same
cache and reuse its fragments.

Each time-series feature has a canonical `dataset:feature` reference. Examples are
`prices:close` and `signals:momentum`. You can use an unqualified name such as `close`
only when one catalog feature has that name. Use an
`output_name -> feature_reference` mapping for output aliases.

Time-series feature selections always cover a half-open interval:

```text
[start, end)
```

Both limits must be timezone-aware ISO 8601 timestamps. The library converts values
with an explicit offset to UTC. The library rejects naive timestamps and date-only
values.

During construction, the library registers each catalog table with DuckDB. The DuckDB
name is the dataset name. If you configure features, the library registers the aligned
selection as `features`. This name is reserved. The selection is a lazy view by default.
With `inmemory=True`, the selection is a materialized table.

## Quick Start

```python
from pymfs import FeatureStore

store = FeatureStore(
    source="data",
    features={
        "price": "prices:close",
        "trend": "signals:sma50",
    },
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    filters={"ticker": ["001", "017"]},
    alignment="point_in_time",
)

result = store.query(
    """
    SELECT ticker, avg(price), avg(trend)
    FROM features
    GROUP BY ticker
    ORDER BY ticker
    """
).fetchall()
```

The library resolves relative `source` and `cache` paths from the current working
directory. It does not change filter values. Each dataset defines its keys and their
format.

## Sources And Caching

Use a local store directly:

```python
store = FeatureStore(source="data")
```

A remote store uses an `hf://datasets/OWNER/NAME` URI or an
`hf://buckets/OWNER/NAME` URI. A remote feature selection requires a local cache:

```python
from os import getenv

store = FeatureStore(
    source="hf://datasets/OWNER/NAME",
    cache="feature_cache",
    token=getenv("HF_TOKEN"),
    features=["prices:close"],
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    alignment="exact",
)
```

The cache downloads only the selected time-series columns, intervals, and filter
scopes. The library treats general tables as small reference data. When you configure a
remote cache, the library downloads each catalog table as a complete snapshot. It
cannot download only selected columns or rows from a catalog table. If a remote catalog
contains multiple tables, construction downloads all the tables. The library registers
each table with DuckDB under its catalog dataset name.

Later selections reuse applicable fragments. The library downloads only missing
coverage. After the cache contains data, you can open it as an offline source:

```python
offline = FeatureStore(source="feature_cache")
```

For an offline read, the library validates the requested features, times, and filters.
If coverage is missing, it raises `FileNotFoundError`. It does not return an incomplete
result. When `token` is `None`, Hugging Face uses its standard authentication chain.

## Selecting Features

Select features during store construction:

```python
store = FeatureStore(
    source="data",
    features=["prices:close", "signals:sma50"],
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    filters={"ticker": ["001"]},
    alignment="exact",
)
```

You cannot change this selection after construction. Supply `features`, `start`, and
`end` together. Create another instance for a different selection:

```python
volume_store = FeatureStore(
  source="data",
  features=["prices:volume"],
  start="2024-01-01T00:00:00Z",
  end="2025-01-01T00:00:00Z",
  alignment="exact",
)
```

Instances can use the same cache directory. The library reuses existing fragments and
downloads only missing coverage. Use SQL predicates to select a smaller interval from
the `features` view. This operation does not change the store. For table-only access,
omit `features`, `start`, and `end` together.

## Alignment

You must set an alignment mode when you configure features.

### Exact

`alignment="exact"` uses inner joins on the declared time column and series keys. Use
this mode when all requested datasets must have values at the same source timestamp.
The datasets must declare compatible time columns and series keys.

Exact alignment deliberately ignores `availability_delay`. For example, a close value
recorded at `08:00` is joined to the `08:00` output row even when its availability
delay is `PT1M`.

Set `alignment="exact"` during `FeatureStore` construction.

### Point In Time

`alignment="point_in_time"` uses as-of joins. A feature value becomes available when:

```text
feature datetime + availability_delay <= output datetime
```

The first requested time-series dataset supplies the output timestamps. For
point-in-time alignment, each requested catalog entry must set `lookahead_safe: true`.
It must also define a valid, non-negative ISO 8601 `availability_delay`. Examples are
`PT0S`, `PT1M`, `PT2H`, and `P1D`. The delay specifies when a causal value is available.
The library cannot verify that the feature calculation was causal.

For example, suppose minute-bar `open` has an availability delay of `PT0S`, while
`close` has a delay of `PT1M`:

| Output time | `open` value | `close` value |
| --- | --- | --- |
| `08:00` | value recorded at `08:00` | previous available value, or null |
| `08:01` | value recorded at `08:01` | value recorded at `08:00` |
| `08:02` | value recorded at `08:02` | value recorded at `08:01` |

Thus, point-in-time alignment prevents the `08:00` close from appearing in the
`08:00` output row because that close was not available until `08:01`. Exact alignment
would place both values in the `08:00` row because it joins their recorded timestamps.

### Alignment Execution

Alignment mode controls how the `features` relation is constructed. The `inmemory`
option controls when that construction is executed:

- With `inmemory=False` (the default), construction registers `features` as a lazy
  DuckDB view. DuckDB reads the Parquet data and performs exact or point-in-time
  alignment when a result is consumed, such as by calling `fetchall()`, `df()`, or
  `to_arrow_table()`.
- With `inmemory=True`, `FeatureStore(...)` immediately reads and aligns the complete
  configured feature selection, then stores it as a reusable in-memory DuckDB table.
  Later queries scan that table.

Remote caching is separate from materialization. Construction can download missing
Parquet fragments into the persistent cache even with `inmemory=False`; the aligned
feature view remains lazy until a query result is consumed.

## Query Interfaces

### SQL

`query()` runs SQL on the shared DuckDB connection. It returns a
`DuckDBPyRelation`. Use a dataset name to address a catalog table. Use `features` to
address the configured time-series selection:

```python
symbols = store.query(
    """
    SELECT ticker, company_name
    FROM symbology
    WHERE ticker = '001'
    """
)

markets = store.query(
    "SELECT market_code, weekday FROM markets"
)

summary = store.query(
  "SELECT ticker, avg(close) FROM features GROUP BY ticker"
)
```

You can use `query()` on a table-only store. You do not have to configure features. To
use `features` in SQL, set `features`, `start`, `end`, and `alignment` during
construction.

### Native DuckDB

`connection()` returns the shared `DuckDBPyConnection`. Use `table()` for native
relation operations. You can also configure DuckDB directly:

```python
connection = store.connection()
features = connection.table("features")
symbols = connection.table("symbology")
connection.execute("SET memory_limit = '4 GiB'")
connection.execute("SET threads = 4")
```

DuckDB scans Parquet when it runs a catalog-table view or a lazy `features` view. With
`inmemory=True`, construction runs the feature selection immediately. It stores the
result in a reusable DuckDB table.

### Outputs And Streaming

DuckDB relations convert directly to Arrow or pandas. Hugging Face Datasets accepts the
Arrow table:

```python
from datasets import Dataset

relation = store.connection().table("features")
table = relation.to_arrow_table()
frame = relation.df()
dataset = Dataset(relation.to_arrow_table())
```

Use the DuckDB Arrow reader to process record batches of a specified size:

```python
for batch in store.connection().table("features").to_arrow_reader(batch_size=250_000):
    process(batch)
```

The batch size limits each Python result allocation. DuckDB controls internal query
memory and disk use.

## Data Specification

### Directory Layout

Use this layout for each local or remote store root:

```text
store/
├── catalog.json
├── ohlcv/
│   ├── metadata.json
│   ├── year=2023/
│   │   └── data.parquet
│   └── year=2024/
│       └── data.parquet
└── symbols/
  ├── metadata.json
  └── data.parquet
```

Each dataset metadata document declares a Parquet `path_template`. A time-series
template contains the `{year}` placeholder. A table template points to one Parquet
file. It must not contain `{year}`. A time-series template can name one file per
partition or use a filename wildcard when a partition contains multiple Parquet files.

```text
ohlcv/year={year}/data.parquet
ohlcv/year={year}/*.parquet
symbols/data.parquet
```

All paths are relative to the store root. Use the same path convention for local
directories, Hugging Face datasets, and Hugging Face buckets.

### Parquet Contract

Each time-series partition must contain these columns:

| Column | Requirement |
| --- | --- |
| Declared `time_column` | Timezone-aware UTC Parquet timestamp identifying the event instant. |
| Declared `series_keys` | Zero or more columns identifying an independent series. |
| Feature columns | One physical column for every catalog feature in the dataset. |

In a time-series dataset, the time column and series keys should identify one row. All
yearly files for a dataset must use compatible schemas. Feature columns can contain
null values. For automatic alignment, all selected datasets must declare the same time
column and series keys.

A table dataset can contain arbitrary columns. The `primary_key` metadata is optional.
Without `primary_key`, the store does not guarantee row identity or uniqueness.

A time-series Parquet file must contain data from only its specified UTC calendar year.
The library uses the `year=YYYY` directory to select files. The Parquet file does not
have to contain a physical year column.

### Catalog Contract

The store root must contain `catalog.json`. Catalog version 3 has this structure:

```json
{
  "catalog_version": 3,
  "name": "owner/store",
  "description": "Research datasets",
  "datasets": [
    {
      "name": "prices",
      "kind": "timeseries",
      "description": "Adjusted prices",
      "metadata": "prices/metadata.json",
      "time_column": "datetime",
      "series_keys": ["ticker"],
      "partitioning": {
        "column": "datetime",
        "unit": "year",
        "timezone": "UTC"
      },
      "min_time": "2023-01-02T08:00:00Z",
      "max_time": "2024-12-31T15:29:00Z"
    },
    {
      "name": "symbology",
      "kind": "table",
      "metadata": "symbols/metadata.json",
      "primary_key": ["ticker"]
    },
    {
      "name": "markets",
      "kind": "table",
      "metadata": "markets/metadata.json"
    }
  ],
  "features": {
    "prices:close": {
      "dataset": "prices",
      "name": "close",
      "metadata": "prices/metadata.json",
      "availability_delay": "PT1M",
      "lookahead_safe": true
    }
  }
}
```

These requirements apply:

- `catalog_version` must be `3`. The library rejects older catalog versions. It does not
  migrate them.
- Every dataset must have kind `timeseries` or `table`.
- The dataset name `features` is reserved and cannot be used by a table dataset.
- A time-series dataset must declare `time_column` and `series_keys`. The list of series
  keys can be empty.
- A table can declare a composite `primary_key`. It can also omit `primary_key`.
- Every feature key must exactly equal `dataset:name` from its entry.
- `datasets[].min_time` and `max_time` are inclusive physical coverage limits. The
  library uses them to limit reads and prevent requests for partitions that do not
  exist.
- Stored instant timestamps and coverage bounds must be timezone-aware UTC values.
- Point-in-time use requires `availability_delay`. The value must be a non-negative,
  calendar-independent ISO 8601 duration. It can contain days, hours, minutes, or
  seconds.
- `lookahead_safe` must be exactly `true` for point-in-time use.
- `metadata` paths are relative to the store root.

### Dataset Metadata Contract

Each dataset should provide `DATASET/metadata.json`. This file gives the detailed
semantic and physical description:

```json
{
  "schema_version": 3,
  "dataset": "prices",
  "kind": "timeseries",
  "description": "Adjusted prices",
  "time_column": "datetime",
  "series_keys": ["ticker"],
  "storage": {
    "format": "parquet",
    "path_template": "prices/year={year}/data.parquet",
    "partition_columns": ["year"]
  },
  "features": {
    "close": {
      "dtype": "float64",
      "unit": "price",
      "description": "Adjusted close price",
      "availability_delay": "PT1M",
      "lookahead_safe": true
    }
  },
  "partitions": [
    {
      "path": "prices/year=2024/data.parquet",
      "rows": 1000000,
      "bytes": 12000000,
      "min_time": "2024-01-02T08:00:00Z",
      "max_time": "2024-12-31T15:29:00Z"
    }
  ]
}
```

Feature metadata can also contain producer-defined fields. Examples are `dependencies`
and `parameters`. The compact catalog is the query-planning index. Dataset metadata
contains more information for discovery, lineage, and validation.

### Cache Contract

A cache is also a readable local store. It contains a copy of `catalog.json`, dataset
metadata, projected time-series fragments, complete table snapshots, and
`manifest.json`.

```json
{
  "version": 1,
  "fragments": [
    {
      "path": "prices/part-20240101T000000-20240201T000000-a1b2c3d4.parquet",
      "dataset": "prices",
      "features": ["close"],
      "start": "2024-01-01T00:00:00+00:00",
      "end": "2024-02-01T00:00:00+00:00",
      "filters": {"ticker": ["001", "017"]}
    },
    {
      "kind": "table",
      "dataset": "symbology",
      "path": "symbols/data.parquet"
    }
  ]
}
```

Time-series fragment intervals are half-open. Each time-series fragment must contain the
current `filters` field. A value of `null` means that the fragment covers all entity
values. The library does not migrate legacy manifest fields such as `tickers`.

When you configure a remote cache, construction copies each catalog table in full. Later
online and offline queries reuse these copies. A table snapshot does not support partial
downloads by column, filter, or row range. DuckDB applies filters and projections when
it reads the local snapshot.

Manifest paths are relative to the cache root. The library updates data files and the
manifest atomically. Thus, a reader does not see a partial document or table file.

## Operational Controls

`show_progress` controls DuckDB progress and completion messages. `disable_cache`
disables the DuckDB external-file, HTTP-metadata, object, and Parquet-metadata caches for
remote reads. Both properties accept Boolean values. A change applies immediately to an
existing connection.

## Limitations

- Reads are for offline or batch operations. The library has no online low-latency
  serving layer.
- A time-series dataset uses a yearly partition template that contains `{year}`.
- Automatically aligned time-series datasets must declare the same `time_column` and
  `series_keys` specifications.
- The first requested time-series dataset defines the point-in-time output timestamps.
- Cache fragments are append-only. The library has no eviction or compaction policy.
- The data model does not include backfill revisions or ingestion timestamps.