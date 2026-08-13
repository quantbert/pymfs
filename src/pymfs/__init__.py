"""Query local or Hugging Face-hosted Parquet feature stores with DuckDB."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, Literal, Sequence, cast
from uuid import uuid4

import duckdb
from huggingface_hub import HfFileSystem


def quote_sql(value: str) -> str:
    """Safely quote a string literal for the small SQL setup statements below."""
    return value.replace("'", "''")


def quote_identifier(value: str) -> str:
    """Quote a DuckDB identifier such as a user-provided output alias."""
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def parse_timestamp(value: str) -> datetime:
    """Parse an aware ISO 8601 timestamp and normalize it to UTC."""
    if not isinstance(value, str):
        raise TypeError("timestamp must be an ISO 8601 string")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid ISO timestamp: {value}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"Timestamp must include a timezone offset: {value}")
    return timestamp.astimezone(UTC)


def parse_availability_delay(value: Any, reference: str) -> timedelta:
    """Parse a non-negative, calendar-independent ISO 8601 duration."""
    if not isinstance(value, str):
        raise ValueError(f"Feature {reference} must define availability_delay")
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
        value,
    )
    if (
        match is None
        or value.endswith("T")
        or not any(part is not None for part in match.groups())
    ):
        raise ValueError(f"Invalid availability_delay for {reference}: {value!r}")
    parts = match.groupdict(default="0")
    return timedelta(
        days=int(parts["days"]),
        hours=int(parts["hours"]),
        minutes=int(parts["minutes"]),
        seconds=float(parts["seconds"]),
    )


class FeatureStore:
    """Thin DuckDB wrapper over feature-family Parquet files on Hugging Face."""

    def __init__(
        self,
        source: Path | str,
        cache: Path | str | None = None,
        token: str | None = None,
        catalog_path: Path | str | None = None,
        features: Sequence[str] | Mapping[str, str] | None = None,
        start: str | None = None,
        end: str | None = None,
        filters: Mapping[str, Sequence[str]] | None = None,
        alignment: Literal["exact", "point_in_time"] | None = None,
        inmemory: bool = False,
    ) -> None:
        if not isinstance(inmemory, bool):
            raise TypeError("inmemory must be a boolean")
        self.token = token
        self.catalog_path = Path(catalog_path) if catalog_path is not None else None
        self._catalog: dict[str, Any] = {}
        self._feature_entries: dict[str, dict[str, Any]] = {}
        self._dataset_entries: dict[str, dict[str, Any]] = {}
        self._catalog_loaded = False
        self._dataset_metadata: dict[str, dict[str, Any]] = {}
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._filesystem: HfFileSystem | None = None
        self._disable_cache = False
        self._show_progress = True
        self._remote_enabled = False
        self.start: datetime | None = None
        self.end: datetime | None = None
        self.filters: dict[str, tuple[str, ...]] | None = None
        self._feature_selection: tuple[str, ...] = ()
        self.alignment = self._validate_alignment(alignment)
        self.inmemory = inmemory

        source_value = str(source).rstrip("/")
        if source_value.startswith("hf://"):
            if not source_value.startswith(("hf://datasets/", "hf://buckets/")):
                raise ValueError("source must be an hf://datasets or hf://buckets URI")
            self._source = source_value
            self._source_path = None
            self._source_is_cache = False
        else:
            source_path = self._local_path(source)
            if not source_path.is_dir():
                raise FileNotFoundError(f"Feature store not found: {source_path}")
            self._source = str(source_path)
            self._source_path = source_path
            self._source_is_cache = (source_path / "manifest.json").is_file()

        self._cache_store = None
        if cache is not None:
            if self._source_path is not None:
                raise ValueError("cache is only supported with a remote source")
            cache_path = self._local_path(cache)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._cache_store = cache_path

        self._cache_tables()

        selection_arguments = (features, start, end)
        if any(value is not None for value in selection_arguments):
            if not all(value is not None for value in selection_arguments):
                raise ValueError("features, start, and end must be provided together")
            assert features is not None and start is not None and end is not None
            self._configure_features(
                features,
                start=start,
                end=end,
                filters=filters,
            )
        elif filters is not None:
            raise ValueError("filters requires features, start, and end")

    @staticmethod
    def _local_path(path: Path | str) -> Path:
        return Path(path).expanduser().resolve()

    @staticmethod
    def _validate_alignment(
        alignment: Literal["exact", "point_in_time"] | None,
    ) -> Literal["exact", "point_in_time"] | None:
        if alignment not in (None, "exact", "point_in_time"):
            raise ValueError("alignment must be 'exact' or 'point_in_time'")
        return alignment

    def _cache_tables(self) -> None:
        self._load_catalog()
        for name, entry in self._dataset_entries.items():
            if entry["kind"] != "table":
                continue
            if name == "features":
                raise ValueError("Table dataset name 'features' is reserved")
            path_template = self._dataset_path_template(name)
            if "{year}" in path_template:
                raise ValueError(
                    f"Table dataset {name!r} cannot use a time-series year path template"
                )
            snapshot_path = self._table_snapshot_path(name, path_template)
            if snapshot_path is not None:
                connection = self._connect()
                path = str(snapshot_path)
            elif self._source_path is None:
                connection = self._enable_remote()
                path = f"{self._source}/{path_template}"
            else:
                connection = self._connect()
                path = str(self._source_path / path_template)
            connection.execute(
                f"CREATE OR REPLACE VIEW {quote_identifier(name)} AS "
                f"SELECT * FROM read_parquet('{quote_sql(path)}')"
            )

    def catalog(self) -> dict[str, Any]:
        """Return the feature catalog used for discovery and query planning."""
        self._load_catalog()
        return self._catalog

    def _load_catalog(self) -> None:
        if self._catalog_loaded:
            return

        candidates: list[Path] = []
        if self.catalog_path is not None:
            candidates.append(self.catalog_path.expanduser())
        if self._cache_store is not None:
            candidates.append(self._cache_store / "catalog.json")
        if self._source_path is not None:
            candidates.append(self._source_path / "catalog.json")

        catalog_path = next((path for path in candidates if path.is_file()), None)
        if catalog_path is None:
            filesystem = self._huggingface_filesystem()
            with filesystem.open(f"{self._source}/catalog.json", "r") as catalog_file:
                catalog_text = catalog_file.read()
        else:
            catalog_text = catalog_path.read_text(encoding="utf-8")

        catalog = json.loads(catalog_text)
        catalog_version = catalog.get("catalog_version")
        if catalog_version != 1:
            raise ValueError(f"Unsupported catalog version: {catalog_version!r}")
        feature_entries = catalog.get("features", {})
        if not isinstance(feature_entries, dict):
            raise ValueError("Catalog features must be a mapping")
        dataset_entries = catalog.get("datasets")
        if not isinstance(dataset_entries, list) or not dataset_entries:
            raise ValueError("Catalog must define at least one dataset")
        dataset_index: dict[str, dict[str, Any]] = {}
        for entry in dataset_entries:
            name = entry.get("name")
            kind = entry.get("kind")
            if not isinstance(name, str) or not name:
                raise ValueError("Catalog datasets must have non-empty names")
            if name in dataset_index:
                raise ValueError(f"Duplicate catalog dataset: {name}")
            if kind not in ("timeseries", "table"):
                raise ValueError(f"Dataset {name!r} has invalid kind: {kind!r}")
            if kind == "timeseries":
                if not isinstance(entry.get("time_column"), str):
                    raise ValueError(
                        f"Timeseries dataset {name!r} requires time_column"
                    )
                if not isinstance(entry.get("series_keys"), list) or not all(
                    isinstance(key, str) and key for key in entry["series_keys"]
                ):
                    raise ValueError(
                        f"Timeseries dataset {name!r} requires series_keys"
                    )
            dataset_index[name] = entry
        for reference, entry in feature_entries.items():
            expected_reference = f"{entry['dataset']}:{entry['name']}"
            if reference != expected_reference:
                raise ValueError(
                    f"Catalog feature {reference!r} must be keyed as {expected_reference!r}"
                )
            dataset = dataset_index.get(entry["dataset"])
            if dataset is None or dataset["kind"] != "timeseries":
                raise ValueError(
                    f"Catalog feature {reference!r} must belong to a timeseries dataset"
                )

        self._catalog = catalog
        self._feature_entries = feature_entries
        self._dataset_entries = dataset_index
        self._catalog_loaded = True
        if (
            self._cache_store is not None
            and catalog_path != self._cache_store / "catalog.json"
        ):
            local_catalog = self._cache_store / "catalog.json"
            temporary_catalog = local_catalog.with_suffix(".json.tmp")
            temporary_catalog.write_text(
                json.dumps(catalog, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_catalog.replace(local_catalog)

    @property
    def cache_path(self) -> Path | None:
        """Configured persistent cache directory, if any."""
        return self._cache_store

    @property
    def source(self) -> str:
        """Configured authoritative feature-store source."""
        return self._source

    @property
    def feature_selection(self) -> tuple[str, ...]:
        """Canonical feature references selected during construction."""
        return self._feature_selection

    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the local DuckDB connection for additional analytical queries."""
        return self._connect()

    @property
    def disable_cache(self) -> bool:
        """Whether DuckDB's local caches are disabled for remote reads."""
        return self._disable_cache

    @disable_cache.setter
    def disable_cache(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("disable_cache must be a boolean")
        self._disable_cache = value
        if self._connection is not None:
            self._apply_cache_settings()

    def _apply_cache_settings(self) -> None:
        connection = self._connection
        if connection is None:
            return
        cache_enabled = "false" if self._disable_cache else "true"
        connection.execute(f"SET enable_external_file_cache = {cache_enabled}")
        if self._disable_cache:
            connection.execute("SET enable_http_metadata_cache = false")
            connection.execute("SET enable_object_cache = false")
            connection.execute("SET parquet_metadata_cache = false")

    @property
    def show_progress(self) -> bool:
        """Whether loads print DuckDB progress and completion statistics."""
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("show_progress must be a boolean")
        self._show_progress = value
        if self._connection is not None:
            self._apply_progress_settings()

    def _apply_progress_settings(self) -> None:
        connection = self._connection
        if connection is None:
            return
        enabled = "true" if self._show_progress else "false"
        connection.execute(f"SET enable_progress_bar = {enabled}")
        connection.execute(f"SET enable_progress_bar_print = {enabled}")
        connection.execute("SET progress_bar_time = 0")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            self._connection = duckdb.connect()
            self._connection.execute("SET TimeZone = 'UTC'")
            self._apply_cache_settings()
            self._apply_progress_settings()
        connection = self._connection
        assert connection is not None
        return connection

    def _enable_remote(self) -> duckdb.DuckDBPyConnection:
        connection = self._connect()
        if not self._remote_enabled:
            connection.register_filesystem(self._huggingface_filesystem())
            self._remote_enabled = True
        return connection

    def _huggingface_filesystem(self) -> HfFileSystem:
        if self._filesystem is None:
            self._filesystem = HfFileSystem(token=self.token)
        return self._filesystem

    def _load_dataset_metadata(self, dataset: str) -> dict[str, Any]:
        """Load and memoize a dataset's detailed metadata document."""
        if dataset in self._dataset_metadata:
            return self._dataset_metadata[dataset]
        entry = self._dataset_entries[dataset]
        metadata_path = entry.get("metadata")
        if not isinstance(metadata_path, str) or not metadata_path:
            raise ValueError(f"Dataset {dataset!r} does not define metadata")
        if self._source_path is not None:
            path = self._source_path / metadata_path
            if not path.is_file():
                raise FileNotFoundError(f"Dataset metadata not found: {path}")
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            filesystem = self._huggingface_filesystem()
            with filesystem.open(
                f"{self._source}/{metadata_path}", "r"
            ) as metadata_file:
                document = json.loads(metadata_file.read())
            if self._cache_store is not None:
                local_metadata = self._cache_store / metadata_path
                local_metadata.parent.mkdir(parents=True, exist_ok=True)
                temporary_metadata = local_metadata.with_suffix(
                    local_metadata.suffix + ".tmp"
                )
                temporary_metadata.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary_metadata.replace(local_metadata)
        self._dataset_metadata[dataset] = document
        return document

    def _dataset_path_template(self, dataset: str) -> str:
        """Return a dataset's Parquet path template relative to the store root."""
        entry = self._dataset_entries[dataset]
        path_template = entry.get("path_template")
        if path_template is None:
            path_template = (
                self._load_dataset_metadata(dataset)
                .get("storage", {})
                .get("path_template")
            )
        if not isinstance(path_template, str) or not path_template:
            raise ValueError(
                f"Dataset {dataset!r} does not define a storage path_template"
            )
        return path_template

    def _read_manifest(self) -> list[dict[str, Any]]:
        store = self._source_path if self._source_is_cache else self._cache_store
        if store is None:
            return []
        manifest_path = store / "manifest.json"
        if not manifest_path.exists():
            return []
        return json.loads(manifest_path.read_text(encoding="utf-8"))["fragments"]

    def _write_manifest(self, fragments: list[dict[str, Any]]) -> None:
        cache_store = self._cache_store
        if cache_store is None:
            raise RuntimeError("Cannot write a manifest without a configured cache")
        manifest_path = cache_store / "manifest.json"
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps({"version": 1, "fragments": fragments}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)

    def _table_snapshot_path(self, dataset: str, path_template: str) -> Path | None:
        """Return or create the complete cached snapshot for a table dataset."""
        if self._source_path is not None and not self._source_is_cache:
            return None

        fragments = self._read_manifest()
        snapshot = next(
            (
                entry
                for entry in fragments
                if entry.get("kind") == "table" and entry.get("dataset") == dataset
            ),
            None,
        )
        store = self._source_path if self._source_is_cache else self._cache_store
        if snapshot is not None and store is not None:
            snapshot_path = store / snapshot["path"]
            if snapshot_path.is_file():
                return snapshot_path

        if self._source_is_cache:
            raise FileNotFoundError(
                f"Local feature store is missing table snapshot for {dataset!r}"
            )
        if self._cache_store is None:
            return None

        destination = self._cache_store / path_template
        temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_path = f"{self._source}/{path_template}"
        if self._show_progress:
            print(f"Caching complete table dataset {dataset!r}...", flush=True)
        temporary_destination.unlink(missing_ok=True)
        try:
            with self._huggingface_filesystem().open(source_path, "rb") as source_file:
                with temporary_destination.open("wb") as destination_file:
                    shutil.copyfileobj(cast(BinaryIO, source_file), destination_file)
            temporary_destination.replace(destination)
        except BaseException:
            temporary_destination.unlink(missing_ok=True)
            raise

        fragments = [
            entry
            for entry in fragments
            if not (entry.get("kind") == "table" and entry.get("dataset") == dataset)
        ]
        fragments.append(
            {
                "kind": "table",
                "dataset": dataset,
                "path": path_template,
            }
        )
        self._write_manifest(fragments)
        return destination

    @staticmethod
    def _covers_filters(
        entry: dict[str, Any], filters: dict[str, list[str]] | None
    ) -> bool:
        cached_filters = entry["filters"]
        if filters is None:
            return cached_filters is None
        if cached_filters is None:
            return True
        return all(
            column in filters and set(filters[column]) <= set(values)
            for column, values in cached_filters.items()
        )

    @staticmethod
    def _missing_intervals(
        start: datetime,
        end: datetime,
        covered: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        clipped = sorted(
            (max(start, covered_start), min(end, covered_end))
            for covered_start, covered_end in covered
            if covered_end > start and covered_start < end
        )
        missing: list[tuple[datetime, datetime]] = []
        cursor = start
        for covered_start, covered_end in clipped:
            if covered_start > cursor:
                missing.append((cursor, covered_start))
            cursor = max(cursor, covered_end)
        if cursor < end:
            missing.append((cursor, end))
        return missing

    @staticmethod
    def _merge_intervals(
        intervals: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        merged: list[list[datetime]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def _remote_paths(self, dataset: str, start: datetime, end: datetime) -> list[str]:
        interval = self._available_interval(dataset, start, end)
        if interval is None:
            return []
        available_start, available_end = interval
        final_timestamp = available_end - timedelta(microseconds=1)
        entry = self._dataset_entries[dataset]
        path_template = entry.get("path_template")
        if path_template is None:
            try:
                path_template = self._dataset_path_template(dataset)
            except (FileNotFoundError, ValueError):
                path_template = f"{dataset}/year={{year}}/data.parquet"
        return [
            f"{self._source}/{path_template.format(year=year)}"
            for year in range(available_start.year, final_timestamp.year + 1)
        ]

    def _available_interval(
        self,
        dataset: str,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, datetime] | None:
        """Clip a half-open interval to a time-series dataset's coverage."""
        dataset_entry = self._dataset_entries.get(dataset)
        if dataset_entry is not None:
            minimum = dataset_entry.get("min_time")
            maximum = dataset_entry.get("max_time")
            if minimum is not None:
                start = max(start, parse_timestamp(minimum))
            if maximum is not None:
                end = min(
                    end,
                    parse_timestamp(maximum) + timedelta(microseconds=1),
                )
        if end <= start:
            return None
        return start, end

    def _cache_features(
        self,
        grouped_features: dict[str, list[str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
    ) -> list[dict[str, Any]]:
        cache_store = self._cache_store
        if cache_store is None:
            raise RuntimeError("Cannot cache features without a configured cache")
        fragments = self._read_manifest()
        for dataset, dataset_features in grouped_features.items():
            dataset_entry = self._dataset_entries[dataset]
            time_column = dataset_entry["time_column"]
            series_keys = dataset_entry["series_keys"]
            interval = self._available_interval(dataset, start, end)
            if interval is None:
                continue
            available_start, available_end = interval
            missing_by_feature: list[tuple[datetime, datetime]] = []
            for feature_name in dataset_features:
                covered = [
                    (parse_timestamp(entry["start"]), parse_timestamp(entry["end"]))
                    for entry in fragments
                    if entry["dataset"] == dataset
                    and feature_name in entry["features"]
                    and self._covers_filters(entry, filters)
                    and (cache_store / entry["path"]).exists()
                ]
                missing_by_feature.extend(
                    self._missing_intervals(available_start, available_end, covered)
                )

            for missing_start, missing_end in self._merge_intervals(missing_by_feature):
                connection = self._enable_remote()
                relative_path = (
                    Path(dataset)
                    / f"part-{missing_start:%Y%m%dT%H%M%S}-{missing_end:%Y%m%dT%H%M%S}"
                    f"-{uuid4().hex[:8]}.parquet"
                )
                destination = cache_store / relative_path
                temporary_destination = destination.with_suffix(".parquet.tmp")
                destination.parent.mkdir(parents=True, exist_ok=True)
                paths = self._remote_paths(dataset, missing_start, missing_end)
                path_sql = ", ".join(f"'{quote_sql(path)}'" for path in paths)
                columns = ", ".join(
                    quote_identifier(column)
                    for column in (time_column, *series_keys, *dataset_features)
                )
                filter_sql = ""
                parameters: list[Any] = [
                    missing_start.isoformat(),
                    missing_end.isoformat(),
                ]
                if filters is not None:
                    for column, values in filters.items():
                        placeholders = ", ".join("?" for _ in values)
                        filter_sql += (
                            f" AND {quote_identifier(column)} IN ({placeholders})"
                        )
                        parameters.extend(values)
                query = (
                    f"COPY (SELECT {columns} FROM read_parquet([{path_sql}]) "
                    f"WHERE {quote_identifier(time_column)} >= CAST(? AS TIMESTAMPTZ) "
                    f"AND {quote_identifier(time_column)} < CAST(? AS TIMESTAMPTZ)"
                    f"{filter_sql}) TO '{quote_sql(str(temporary_destination))}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                if self._show_progress:
                    print(
                        f"Caching features from {dataset!r}: "
                        f"{', '.join(dataset_features)} for "
                        f"[{missing_start.isoformat()}, {missing_end.isoformat()})...",
                        flush=True,
                    )
                temporary_destination.unlink(missing_ok=True)
                try:
                    connection.execute(query, parameters)
                    temporary_destination.replace(destination)
                except BaseException:
                    temporary_destination.unlink(missing_ok=True)
                    raise
                fragments.append(
                    {
                        "path": relative_path.as_posix(),
                        "dataset": dataset,
                        "features": dataset_features,
                        "start": missing_start.isoformat(),
                        "end": missing_end.isoformat(),
                        "filters": filters,
                    }
                )
                self._write_manifest(fragments)
        return fragments

    def _validate_offline_coverage(
        self,
        fragments: list[dict[str, Any]],
        grouped_features: dict[str, list[str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
        store: Path | None = None,
    ) -> None:
        fragment_store = self._source_path if store is None else store
        for dataset, dataset_features in grouped_features.items():
            interval = self._available_interval(dataset, start, end)
            if interval is None:
                continue
            available_start, available_end = interval
            for feature_name in dataset_features:
                covered = [
                    (parse_timestamp(entry["start"]), parse_timestamp(entry["end"]))
                    for entry in fragments
                    if entry["dataset"] == dataset
                    and feature_name in entry["features"]
                    and self._covers_filters(entry, filters)
                    and (fragment_store / entry["path"]).exists()
                ]
                missing = self._missing_intervals(
                    available_start, available_end, covered
                )
                if missing:
                    intervals = ", ".join(
                        f"[{missing_start.isoformat()}, {missing_end.isoformat()})"
                        for missing_start, missing_end in missing
                    )
                    raise FileNotFoundError(
                        f"Local feature store is missing coverage for {feature_name}: "
                        f"{intervals}"
                    )

    def _resolve_features(
        self,
        requested_features: Sequence[str] | Mapping[str, str],
    ) -> list[tuple[str, str, str, str]]:
        if not requested_features:
            raise ValueError("At least one feature must be requested")
        self._load_catalog()
        feature_entries = self._feature_entries

        if isinstance(requested_features, str):
            raise TypeError("requested_features must be a sequence or alias mapping")
        if isinstance(requested_features, Mapping):
            requested_items = list(requested_features.items())
        else:
            requested_items = [(None, reference) for reference in requested_features]

        expanded_items: list[tuple[str | None, str]] = []
        for requested_alias, requested_reference in requested_items:
            if not isinstance(requested_reference, str) or not requested_reference:
                raise ValueError("Feature references must be non-empty strings")
            if requested_reference.endswith(":*"):
                if requested_alias is not None:
                    raise ValueError("Feature wildcards cannot have output aliases")
                prefix = requested_reference[:-1]
                matches = [
                    reference
                    for reference in feature_entries
                    if reference.startswith(prefix)
                ]
                if not matches:
                    raise ValueError(f"Unknown feature group: {requested_reference}")
                expanded_items.extend((None, reference) for reference in matches)
            else:
                expanded_items.append((requested_alias, requested_reference))

        resolved_features: list[tuple[str, str, str, str]] = []
        seen_requests: set[tuple[str, str]] = set()
        for requested_alias, requested_reference in expanded_items:
            reference = requested_reference
            if ":" not in reference:
                matches = [
                    candidate
                    for candidate, entry in feature_entries.items()
                    if entry["name"] == reference
                ]
                if not matches:
                    raise ValueError(f"Unknown feature: {reference}")
                if len(matches) > 1:
                    raise ValueError(
                        f"Ambiguous feature {reference!r}; use one of: {', '.join(matches)}"
                    )
                reference = matches[0]
            if reference not in feature_entries:
                raise ValueError(
                    f"Unknown feature: {reference}. "
                    f"Available: {', '.join(feature_entries)}"
                )

            entry = feature_entries[reference]
            output_name = requested_alias or entry["name"]
            if not isinstance(output_name, str) or not output_name:
                raise ValueError("Output aliases must be non-empty strings")
            dataset_entry = self._dataset_entries[entry["dataset"]]
            reserved_columns = {
                dataset_entry["time_column"],
                *dataset_entry["series_keys"],
            }
            if output_name in reserved_columns:
                raise ValueError(f"Output alias is reserved: {output_name}")
            request_key = (output_name, reference)
            if request_key not in seen_requests:
                resolved_features.append(
                    (output_name, reference, entry["dataset"], entry["name"])
                )
                seen_requests.add(request_key)

        output_names = [output_name for output_name, *_ in resolved_features]
        duplicate_outputs = sorted(
            name for name in set(output_names) if output_names.count(name) > 1
        )
        if duplicate_outputs:
            raise ValueError(
                f"Duplicate output columns: {', '.join(duplicate_outputs)}. "
                "Use an alias mapping to give them distinct names."
            )
        return resolved_features

    @staticmethod
    def _normalize_filters(
        filters: Mapping[str, Sequence[str]] | None,
    ) -> dict[str, list[str]] | None:
        if filters is None:
            return None
        if not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping of columns to string values")
        if not filters:
            raise ValueError("filters cannot be empty")
        normalized_filters: dict[str, list[str]] = {}
        for column, values in filters.items():
            if not isinstance(column, str) or not column:
                raise ValueError("filter columns must be non-empty strings")
            if isinstance(values, str) or not isinstance(values, Sequence):
                raise TypeError(f"filter values for {column!r} must be a sequence")
            normalized_values: list[str] = []
            for value in values:
                if not isinstance(value, str):
                    raise TypeError(f"filter values for {column!r} must be strings")
                normalized_values.append(value)
            if not normalized_values:
                raise ValueError(f"filter values for {column!r} cannot be empty")
            normalized_filters[column] = normalized_values
        return normalized_filters

    @staticmethod
    def _group_features(
        resolved_features: list[tuple[str, str, str, str]],
    ) -> dict[str, list[str]]:
        grouped_features: dict[str, list[str]] = {}
        for _, _, dataset, feature_name in resolved_features:
            dataset_features = grouped_features.setdefault(dataset, [])
            if feature_name not in dataset_features:
                dataset_features.append(feature_name)
        return grouped_features

    def _timeseries_keys(
        self, grouped_features: dict[str, list[str]]
    ) -> tuple[str, list[str]]:
        """Return the common time and series keys required for automatic alignment."""
        specifications = {
            (
                self._dataset_entries[dataset]["time_column"],
                tuple(self._dataset_entries[dataset]["series_keys"]),
            )
            for dataset in grouped_features
        }
        if len(specifications) != 1:
            details = ", ".join(
                f"{dataset}=(time_column={self._dataset_entries[dataset]['time_column']!r}, "
                f"series_keys={self._dataset_entries[dataset]['series_keys']!r})"
                for dataset in grouped_features
            )
            raise ValueError(
                "Automatic alignment requires compatible time_column and series_keys: "
                f"{details}"
            )
        time_column, series_keys = specifications.pop()
        return time_column, list(series_keys)

    def _availability_delays(
        self,
        resolved_features: list[tuple[str, str, str, str]],
        *,
        require_safe: bool,
    ) -> dict[str, timedelta]:
        availability_delays: dict[str, timedelta] = {}
        for _, reference, _, _ in resolved_features:
            entry = self._feature_entries[reference]
            if require_safe and entry.get("lookahead_safe") is not True:
                raise ValueError(f"Feature {reference} is not marked lookahead_safe")
            value = entry.get("availability_delay")
            if value is not None:
                availability_delays[reference] = parse_availability_delay(
                    value, reference
                )
            elif require_safe:
                raise ValueError(f"Feature {reference} must define availability_delay")
            else:
                availability_delays[reference] = timedelta()
        return availability_delays

    def _configure_features(
        self,
        requested_features: Sequence[str] | Mapping[str, str],
        *,
        start: str,
        end: str,
        filters: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Configure the immutable feature slice and ensure its coverage is local."""
        if self.alignment is None:
            raise TypeError("alignment is required when features are configured")
        alignment = cast(Literal["exact", "point_in_time"], self.alignment)
        resolved_features = self._resolve_features(requested_features)
        start_timestamp = parse_timestamp(start)
        end_timestamp = parse_timestamp(end)
        if end_timestamp <= start_timestamp:
            raise ValueError("end must be later than start")
        normalized_filters = self._normalize_filters(filters)
        grouped_features = self._group_features(resolved_features)
        physical_start = start_timestamp
        if self.alignment == "point_in_time":
            delays = self._availability_delays(resolved_features, require_safe=False)
            physical_start -= max(delays.values(), default=timedelta())

        if self._source_path is None and self._cache_store is None:
            raise ValueError(
                "remote feature selection requires a configured local cache"
            )
        if self._source_is_cache:
            self._validate_offline_coverage(
                self._read_manifest(),
                grouped_features,
                physical_start,
                end_timestamp,
                normalized_filters,
            )
        elif self._cache_store is not None:
            self._cache_features(
                grouped_features,
                physical_start,
                end_timestamp,
                normalized_filters,
            )

        relation = self._relation(
            resolved_features,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            filters=normalized_filters,
            alignment=alignment,
        )
        if self.inmemory:
            if self.show_progress:
                print(
                    f"Materializing reusable {self.alignment} features for "
                    f"[{start_timestamp.isoformat()}, {end_timestamp.isoformat()})...",
                    flush=True,
                )
            started_at = perf_counter()
            relation.create("features")
            if self.show_progress:
                print(
                    f"Materialized features inside DuckDB in "
                    f"{perf_counter() - started_at:.2f}s",
                    flush=True,
                )
        else:
            relation.create_view("features", replace=True)

        self.start = start_timestamp
        self.end = end_timestamp
        self.filters = (
            None
            if normalized_filters is None
            else {
                column: tuple(values) for column, values in normalized_filters.items()
            }
        )
        self._feature_selection = tuple(
            reference for _, reference, _, _ in resolved_features
        )

    def _relation(
        self,
        resolved_features: list[tuple[str, str, str, str]],
        *,
        start_timestamp: datetime,
        end_timestamp: datetime,
        filters: dict[str, list[str]] | None,
        alignment: Literal["exact", "point_in_time"],
    ) -> duckdb.DuckDBPyRelation:
        """Build the native lazy DuckDB relation for the configured feature slice."""
        if alignment not in ("exact", "point_in_time"):
            raise ValueError("alignment must be 'exact' or 'point_in_time'")

        availability_delays: dict[str, timedelta] = {}
        if alignment == "point_in_time":
            availability_delays = self._availability_delays(
                resolved_features, require_safe=True
            )
        maximum_delay = max(availability_delays.values(), default=timedelta())
        read_start = start_timestamp - maximum_delay
        grouped_features = self._group_features(resolved_features)
        time_column, series_keys = self._timeseries_keys(grouped_features)
        key_columns = [time_column, *series_keys]
        quoted_key_columns = [quote_identifier(column) for column in key_columns]
        dataset_aliases = {
            dataset: f"family_{index}" for index, dataset in enumerate(grouped_features)
        }

        if self._source_is_cache:
            fragments = self._read_manifest()
            self._validate_offline_coverage(
                fragments,
                grouped_features,
                read_start,
                end_timestamp,
                filters,
            )
        elif self._cache_store is not None:
            fragments = self._read_manifest()
            self._validate_cache_coverage(
                fragments, grouped_features, read_start, end_timestamp, filters
            )
        else:
            fragments = []
        fragment_store = (
            self._source_path if self._source_is_cache else self._cache_store
        )

        read_start_sql = quote_sql(read_start.isoformat())
        end_sql = quote_sql(end_timestamp.isoformat())
        ctes: list[str] = []
        for index, (dataset, dataset_features) in enumerate(grouped_features.items()):
            dataset_time_column = self._dataset_entries[dataset]["time_column"]
            common_filters = (
                f"{quote_identifier(dataset_time_column)} >= TIMESTAMPTZ '{read_start_sql}' "
                f"AND {quote_identifier(dataset_time_column)} < TIMESTAMPTZ '{end_sql}'"
            )
            if not fragments:
                paths = self._remote_paths(dataset, read_start, end_timestamp)
            else:
                assert fragment_store is not None
                paths = sorted(
                    {
                        str(fragment_store / entry["path"])
                        for entry in fragments
                        if entry["dataset"] == dataset
                        and set(entry.get("features", ())) & set(dataset_features)
                        and self._covers_filters(entry, filters)
                        and (fragment_store / entry["path"]).exists()
                        and parse_timestamp(entry["end"]) > read_start
                        and parse_timestamp(entry["start"]) < end_timestamp
                    }
                )
            empty_family = not paths
            if empty_family:
                if fragments:
                    paths = sorted(
                        {
                            str(fragment_store / entry["path"])
                            for entry in fragments
                            if entry["dataset"] == dataset
                            and set(dataset_features) <= set(entry.get("features", ()))
                            and self._covers_filters(entry, filters)
                            and (fragment_store / entry["path"]).exists()
                        }
                    )
                elif self._source_path is not None:
                    path_template = self._dataset_path_template(dataset)
                    paths = sorted(
                        str(path)
                        for path in self._source_path.glob(
                            path_template.format(year="*")
                        )
                        if path.is_file()
                    )
                else:
                    minimum = self._dataset_entries[dataset].get("min_time")
                    if minimum is not None:
                        minimum_timestamp = parse_timestamp(minimum)
                        paths = self._remote_paths(
                            dataset,
                            minimum_timestamp,
                            minimum_timestamp + timedelta(microseconds=1),
                        )
                        self._enable_remote()
                if not paths:
                    raise FileNotFoundError(
                        f"Cannot determine schema for out-of-range dataset {dataset!r}"
                    )
            path_sql = ", ".join(f"'{quote_sql(path)}'" for path in paths)
            filter_sql = ""
            if filters is not None:
                for column, values in filters.items():
                    filter_values = ", ".join(
                        f"'{quote_sql(value)}'" for value in values
                    )
                    filter_sql += (
                        f" AND {quote_identifier(column)} IN ({filter_values})"
                    )
            columns = ", ".join(
                quote_identifier(column) for column in (*key_columns, *dataset_features)
            )
            if empty_family:
                ctes.append(
                    f"family_{index} AS (SELECT {columns} "
                    f"FROM read_parquet([{path_sql}], union_by_name=true) WHERE false)"
                )
            elif not fragments or len(paths) == 1:
                ctes.append(
                    f"family_{index} AS (SELECT {columns} FROM read_parquet([{path_sql}]) "
                    f"WHERE {common_filters}{filter_sql})"
                )
            else:
                aggregates = ", ".join(
                    f"max({quote_identifier(feature_name)}) AS {quote_identifier(feature_name)}"
                    for feature_name in dataset_features
                )
                ctes.append(
                    f"family_{index} AS (SELECT {', '.join(quoted_key_columns)}, {aggregates} "
                    f"FROM read_parquet([{path_sql}], union_by_name=true) "
                    f"WHERE {common_filters}{filter_sql} "
                    f"GROUP BY {', '.join(quoted_key_columns)})"
                )

        if alignment == "exact":
            select_columns = [f"family_0.{column}" for column in quoted_key_columns]
            select_columns.extend(
                f"{dataset_aliases[dataset]}.{quote_identifier(feature_name)} "
                f"AS {quote_identifier(output_name)}"
                for output_name, _, dataset, feature_name in resolved_features
            )
            joins = [
                f"JOIN family_{index} USING ({', '.join(quoted_key_columns)})"
                for index in range(1, len(grouped_features))
            ]
            query = (
                f"WITH {', '.join(ctes)} SELECT {', '.join(select_columns)} "
                f"FROM family_0 {' '.join(joins)}"
            )
        else:
            start_sql = quote_sql(start_timestamp.isoformat())
            ctes.append(
                f"spine AS (SELECT {', '.join(quoted_key_columns)} FROM family_0 "
                f"WHERE {quote_identifier(time_column)} >= TIMESTAMPTZ '{start_sql}' "
                f"AND {quote_identifier(time_column)} < TIMESTAMPTZ '{end_sql}')"
            )
            alignment_groups: dict[tuple[str, int], str] = {}
            for _, reference, dataset, _ in resolved_features:
                delay_microseconds = availability_delays[reference] // timedelta(
                    microseconds=1
                )
                alignment_groups.setdefault(
                    (dataset, delay_microseconds),
                    f"aligned_{len(alignment_groups)}",
                )
            select_columns = [f"spine.{column}" for column in quoted_key_columns]
            select_columns.extend(
                f"{alignment_groups[(dataset, availability_delays[reference] // timedelta(microseconds=1))]}."
                f"{quote_identifier(feature_name)} AS {quote_identifier(output_name)}"
                for output_name, reference, dataset, feature_name in resolved_features
            )
            joins = []
            for (dataset, delay_microseconds), alias in alignment_groups.items():
                key_predicates = [
                    f"spine.{quote_identifier(key)} = {alias}.{quote_identifier(key)}"
                    for key in series_keys
                ]
                key_predicates.append(
                    f"spine.{quote_identifier(time_column)} >= "
                    f"{alias}.{quote_identifier(time_column)} "
                    f"+ INTERVAL {delay_microseconds} MICROSECOND"
                )
                joins.append(
                    f"ASOF LEFT JOIN {dataset_aliases[dataset]} AS {alias} "
                    f"ON {' AND '.join(key_predicates)}"
                )
            query = (
                f"WITH {', '.join(ctes)} SELECT {', '.join(select_columns)} "
                f"FROM spine {' '.join(joins)}"
            )
        return self._connect().sql(query)

    def _validate_cache_coverage(
        self,
        fragments: list[dict[str, Any]],
        grouped_features: dict[str, list[str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
    ) -> None:
        self._validate_offline_coverage(
            fragments,
            grouped_features,
            start,
            end,
            filters,
            store=self._cache_store,
        )

    def query(self, sql: str) -> duckdb.DuckDBPyRelation:
        """Query registered catalog tables and the optional active feature slice."""
        return self._connect().sql(sql)

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        """Return a registered catalog table as a lazy DuckDB relation."""
        if not isinstance(name, str) or not name:
            raise ValueError("table name must be a non-empty string")
        return self._connect().table(name)

    def features(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        columns: Sequence[str] | None = None,
        order_by: Sequence[str] | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Return the configured feature selection as a lazy DuckDB relation."""
        if not self._feature_selection:
            raise ValueError("FeatureStore has no configured feature selection")

        relation = self.table("features")
        grouped_features = self._group_features(
            self._resolve_features(self._feature_selection)
        )
        time_column, _ = self._timeseries_keys(grouped_features)
        predicates: list[str] = []
        if start is not None:
            start_timestamp = parse_timestamp(start)
            predicates.append(
                f"{quote_identifier(time_column)} >= "
                f"TIMESTAMPTZ '{quote_sql(start_timestamp.isoformat())}'"
            )
        if end is not None:
            end_timestamp = parse_timestamp(end)
            predicates.append(
                f"{quote_identifier(time_column)} < "
                f"TIMESTAMPTZ '{quote_sql(end_timestamp.isoformat())}'"
            )
        if predicates:
            relation = relation.filter(" AND ".join(predicates))

        if columns is not None:
            selected_columns = self._relation_columns(columns, "columns")
            relation = relation.project(
                ", ".join(quote_identifier(column) for column in selected_columns)
            )
        if order_by is not None:
            ordering_columns = self._relation_columns(order_by, "order_by")
            relation = relation.order(
                ", ".join(quote_identifier(column) for column in ordering_columns)
            )
        return relation

    def feature_batches(
        self,
        *,
        window: timedelta,
        start: str | None = None,
        end: str | None = None,
        columns: Sequence[str] | None = None,
        order_by: Sequence[str] | None = None,
    ) -> Iterator[duckdb.DuckDBPyRelation]:
        """Yield lazy feature relations over consecutive time windows."""
        if not self._feature_selection:
            raise ValueError("FeatureStore has no configured feature selection")
        if not isinstance(window, timedelta):
            raise TypeError("window must be a datetime.timedelta")
        if window <= timedelta():
            raise ValueError("window must be positive")

        start_timestamp = parse_timestamp(start) if start is not None else self.start
        end_timestamp = parse_timestamp(end) if end is not None else self.end
        assert start_timestamp is not None and end_timestamp is not None
        if end_timestamp <= start_timestamp:
            raise ValueError("end must be later than start")

        batch_start = start_timestamp
        while batch_start < end_timestamp:
            batch_end = min(batch_start + window, end_timestamp)
            yield self.features(
                start=batch_start.isoformat(),
                end=batch_end.isoformat(),
                columns=columns,
                order_by=order_by,
            )
            batch_start = batch_end

    @staticmethod
    def _relation_columns(columns: Sequence[str], argument: str) -> tuple[str, ...]:
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise TypeError(f"{argument} must be a sequence of column names")
        normalized = tuple(columns)
        if not normalized or not all(
            isinstance(column, str) and column for column in normalized
        ):
            raise ValueError(f"{argument} must contain non-empty column names")
        return normalized
