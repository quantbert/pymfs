"""Build the dataset catalog, metadata documents, and storage README."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


CATALOG_VERSION = 3
DATASETS: dict[str, dict[str, Any]] = {
    "ohlcv": {
        "kind": "timeseries",
        "description": "Synthetic minute OHLCV bars for Nasdaq Stockholm sessions.",
        "directory": "ohlcv",
        "path_template": "ohlcv/year={year}/data.parquet",
        "time_column": "datetime",
        "series_keys": ["ticker"],
        "features": {
            "open": {
                "dtype": "float64",
                "unit": "price",
                "description": "Bar open price.",
                "availability_delay": "PT0S",
                "lookahead_safe": True,
            },
            **{
                name: {
                    "dtype": dtype,
                    "unit": unit,
                    "description": description,
                    "availability_delay": "PT1M",
                    "lookahead_safe": True,
                }
                for name, dtype, unit, description in (
                    ("high", "float64", "price", "Bar high price."),
                    ("low", "float64", "price", "Bar low price."),
                    ("close", "float64", "price", "Bar close price."),
                    ("volume", "int64", "shares", "Bar traded volume."),
                )
            },
        },
    },
    "sma": {
        "kind": "timeseries",
        "description": "Simple moving averages calculated from minute close prices.",
        "directory": "sma",
        "path_template": "sma/year={year}/data.parquet",
        "time_column": "datetime",
        "series_keys": ["ticker"],
        "features": {
            f"sma{window}": {
                "dtype": "float64",
                "unit": "price",
                "description": f"Trailing {window}-bar simple moving average of close.",
                "dependencies": ["ohlcv:close"],
                "parameters": {"window": window},
                "lookahead_safe": True,
                "availability_delay": "PT1M",
            }
            for window in (10, 20, 50, 200)
        },
    },
    "symbology": {
        "kind": "table",
        "description": "Synthetic security identifiers and company information.",
        "directory": "symbols",
        "path_template": "symbols/data.parquet",
        "primary_key": ["ticker"],
        "columns": {
            "ticker": {"dtype": "string", "description": "Synthetic ticker symbol."},
            "isin": {"dtype": "string", "description": "Synthetic ISIN."},
            "cik": {"dtype": "string", "description": "Synthetic CIK."},
            "company_name": {"dtype": "string", "description": "Synthetic company name."},
            "description": {"dtype": "string", "description": "Synthetic company description."},
            "market_code": {"dtype": "string", "description": "Listing market MIC."},
        },
    },
    "markets": {
        "kind": "table",
        "description": "Illustrative weekly trading hours by market code.",
        "directory": "markets",
        "path_template": "markets/data.parquet",
        "columns": {
            "market_code": {"dtype": "string", "description": "Market identifier code."},
            "weekday": {"dtype": "string", "description": "Weekday for these hours."},
            "opens_at": {"dtype": "time32", "description": "Local market opening time."},
            "closes_at": {"dtype": "time32", "description": "Local market closing time."},
            "timezone": {"dtype": "string", "description": "IANA timezone for the local times."},
        },
    },
}


def json_value(value: Any) -> Any:
    """Convert Parquet statistic values to JSON-compatible values."""
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("Parquet timestamp statistics must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def inspect_partition(
    path: Path, data_root: Path, time_column: str | None = None
) -> dict[str, Any]:
    """Read row counts and datetime bounds without scanning Parquet data pages."""
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    datetime_index = schema.get_field_index(time_column) if time_column else -1
    if time_column:
        if datetime_index < 0:
            raise ValueError(f"{path} does not contain time column {time_column!r}")
        timestamp_type = schema.field(datetime_index).type
        if not hasattr(timestamp_type, "tz") or timestamp_type.tz != "UTC":
            raise ValueError(
                f"{path} time column {time_column!r} must be a timezone-aware UTC timestamp"
            )
    minimums: list[Any] = []
    maximums: list[Any] = []
    if datetime_index >= 0:
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            statistics = parquet_file.metadata.row_group(row_group_index).column(
                datetime_index
            ).statistics
            if statistics is not None and statistics.has_min_max:
                minimums.append(statistics.min)
                maximums.append(statistics.max)

    partition: dict[str, Any] = {
        "path": path.relative_to(data_root).as_posix(),
        "rows": parquet_file.metadata.num_rows,
        "bytes": path.stat().st_size,
    }
    if minimums:
        partition["min_time"] = json_value(min(minimums))
        partition["max_time"] = json_value(max(maximums))
    return partition


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically write a deterministic JSON document."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_text(path: Path, content: str) -> None:
    """Atomically write a UTF-8 text artifact."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def human_size(byte_count: int) -> str:
    """Format a byte count for the human-readable store README."""
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def render_store_readme(dataset_metadata: list[dict[str, Any]], source: str) -> str:
    """Render the README shown on the Hugging Face storage page."""
    config_lines: list[str] = []
    for index, metadata in enumerate(dataset_metadata):
        dataset_name = metadata["dataset"]
        data_path = metadata["storage"]["path_template"].replace("{year}", "*")
        config_lines.extend(
            [
                f"- config_name: {dataset_name}",
                *(["  default: true"] if index == 0 else []),
                "  data_files:",
                "  - split: train",
                f'    path: "{data_path}"',
            ]
        )

    sections: list[str] = []
    for metadata in dataset_metadata:
        partitions = metadata["partitions"]
        row_count = sum(partition["rows"] for partition in partitions)
        byte_count = sum(partition["bytes"] for partition in partitions)
        definitions = metadata.get("features", metadata.get("columns", {}))
        column_rows = [
            f"| `{name}` | `{definition['dtype']}` | {definition['description']} |"
            for name, definition in definitions.items()
        ]
        if metadata["kind"] == "timeseries":
            minimum = min(partition["min_time"] for partition in partitions)
            maximum = max(partition["max_time"] for partition in partitions)
            coverage = (
                f"Coverage: `{minimum}` through `{maximum}`. "
                f"{row_count:,} rows across {len(partitions)} UTC-year partitions "
                f"({human_size(byte_count)} compressed)."
            )
        else:
            key_description = (
                f"Primary key: `{', '.join(metadata['primary_key'])}`."
                if "primary_key" in metadata
                else "No primary key is declared."
            )
            coverage = (
                f"{row_count:,} rows ({human_size(byte_count)} compressed). "
                f"{key_description}"
            )
        sections.append(
            "\n".join(
                [
                    f"### {metadata['dataset']}",
                    "",
                    metadata["description"],
                    "",
                    coverage,
                    "",
                    "| Column | Type | Description |",
                    "| --- | --- | --- |",
                    *column_rows,
                ]
            )
        )

    return (
        "---\n"
        "pretty_name: Hifinab Research Feature Store\n"
        "tags:\n"
        "- timeseries\n"
        "- finance\n"
        "- tabular\n"
        "- parquet\n"
        "- duckdb\n"
        "configs:\n"
        + "\n".join(config_lines)
        + "\n---\n\n"
        "# Hifinab Research Feature Store\n\n"
        "Private Parquet dataset storage for research and model-training data "
        "preparation. Instant timestamps are stored as timezone-aware UTC values.\n\n"
        "> This store currently contains deterministic synthetic market data. "
        "It does not represent real trades or prices and must not be used for live "
        "trading or investment decisions.\n\n"
        "## Store Structure\n\n"
        "Datasets have independent schemas and may be time series or general tables. "
        "Time-series data is partitioned by UTC year.\n\n"
        + "\n\n".join(sections)
        + "\n\n## Loading\n\n"
        "The project feature-store wrapper resolves features through `catalog.json`, "
        "downloads only relevant yearly partitions, and joins datasets locally with "
        "DuckDB:\n\n"
        "```python\n"
        "from os import getenv\n"
        "\n"
        "from pymfs import FeatureStore\n"
        "\n"
        "features = FeatureStore(\n"
        f"    source=\"{source}\",\n"
        "    cache=\"feature_cache\",\n"
        "    token=getenv(\"HF_TOKEN\"),\n"
        "    features={\"close\": \"ohlcv:close\", \"long_average\": \"sma:sma200\"},\n"
        "    start=\"2024-01-01T00:00:00Z\",\n"
        "    end=\"2025-01-01T00:00:00Z\",\n"
        "    alignment=\"exact\",\n"
        ")\n"
        "frame = features.query(\"SELECT * FROM features\").df()\n"
        "\n"
        "# Align each value to the first timestamp when it was safely available.\n"
        "training_data = FeatureStore(\n"
        f"    source=\"{source}\",\n"
        "    cache=\"feature_cache\",\n"
        "    token=getenv(\"HF_TOKEN\"),\n"
        "    features=[\"ohlcv:close\", \"sma:sma200\"],\n"
        "    start=\"2024-01-01T00:00:00Z\",\n"
        "    end=\"2025-01-01T00:00:00Z\",\n"
        "    alignment=\"point_in_time\",\n"
        ")\n"
        "training_relation = training_data.connection().table(\"features\")\n"
        "\n"
        "# Use a populated cache without network access.\n"
        "offline_features = FeatureStore(source=\"feature_cache\")\n"
        "```\n\n"
        "See `catalog.json` for dataset discovery and each dataset's `metadata.json` for "
        "semantic definitions, dependencies, storage layout, and partition statistics.\n\n"
        "## Intended Use\n\n"
        "The dataset is intended for testing research pipelines, feature selection, "
        "training-data compilation, caching, and local analytical workflows. Results "
        "obtained from the synthetic values are not evidence of real-world model "
        "performance.\n"
    )


def build_catalog(data_root: Path, store_name: str, source: str) -> dict[str, Any]:
    """Write the version 3 catalog, dataset metadata, and storage README."""
    dataset_entries: list[dict[str, Any]] = []
    dataset_metadata: list[dict[str, Any]] = []
    feature_index: dict[str, dict[str, Any]] = {}

    for dataset_name, definition in DATASETS.items():
        dataset_root = data_root / definition["directory"]
        if definition["kind"] == "timeseries":
            paths = sorted(dataset_root.glob("year=*/*.parquet"))
        else:
            paths = [data_root / definition["path_template"]]
            paths = [path for path in paths if path.is_file()]
        partitions = [
            inspect_partition(path, data_root, definition.get("time_column"))
            for path in paths
        ]
        if not partitions:
            raise FileNotFoundError(f"No Parquet data found for dataset {dataset_name!r}")

        metadata_path = dataset_root / "metadata.json"
        metadata = {
            "schema_version": CATALOG_VERSION,
            "dataset": dataset_name,
            "kind": definition["kind"],
            "description": definition["description"],
            "storage": {
                "format": "parquet",
                "path_template": definition["path_template"],
                "partition_columns": ["year"]
                if definition["kind"] == "timeseries"
                else [],
            },
            "partitions": partitions,
        }
        for field in ("time_column", "series_keys", "primary_key", "features", "columns"):
            if field in definition:
                metadata[field] = definition[field]
        write_json(metadata_path, metadata)
        dataset_metadata.append(metadata)

        entry = {
            "name": dataset_name,
            "kind": definition["kind"],
            "description": definition["description"],
            "metadata": metadata_path.relative_to(data_root).as_posix(),
        }
        for field in ("time_column", "series_keys", "primary_key"):
            if field in definition:
                entry[field] = definition[field]
        if definition["kind"] == "timeseries":
            entry["partitioning"] = {
                "column": definition["time_column"],
                "unit": "year",
                "timezone": "UTC",
            }
            entry["min_time"] = min(partition["min_time"] for partition in partitions)
            entry["max_time"] = max(partition["max_time"] for partition in partitions)
            for feature_name, feature_definition in definition["features"].items():
                feature_reference = f"{dataset_name}:{feature_name}"
                feature_index[feature_reference] = {
                    "dataset": dataset_name,
                    "name": feature_name,
                    "metadata": metadata_path.relative_to(data_root).as_posix(),
                    "availability_delay": feature_definition["availability_delay"],
                    "lookahead_safe": feature_definition["lookahead_safe"],
                }
        dataset_entries.append(entry)

    catalog = {
        "catalog_version": CATALOG_VERSION,
        "name": store_name,
        "description": "Research and model-training datasets.",
        "datasets": dataset_entries,
        "features": feature_index,
    }
    write_json(data_root / "catalog.json", catalog)
    write_text(data_root / "README.md", render_store_readme(dataset_metadata, source))
    return catalog


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--store-name", required=True)
    parser.add_argument("--source", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    settings = arguments()
    catalog = build_catalog(
        settings.data.resolve(),
        store_name=settings.store_name,
        source=settings.source,
    )
    print(
        f"Wrote metadata and store README for {len(catalog['datasets'])} datasets and "
        f"{len(catalog['features'])} features to {settings.data.resolve()}"
    )