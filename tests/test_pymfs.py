import json
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from datasets import Dataset

from pymfs import FeatureStore, parse_availability_delay


LOAD_ARGUMENTS = {
    "start": "2024-01-02T08:00:00Z",
    "end": "2024-01-02T08:02:00Z",
    "filters": {"ticker": ["001"]},
}


class FeatureStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.fixture_path = Path(cls.fixture_directory.name)
        timestamps = [
            datetime(2024, 1, 2, 8, minute, tzinfo=UTC) for minute in range(3)
        ]
        cls._write_family(
            "ohlcv",
            pa.table(
                {
                    "datetime": timestamps,
                    "ticker": ["001"] * 3,
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1000, 1100, 1200],
                }
            ),
        )
        cls._write_family(
            "sma",
            pa.table(
                {
                    "datetime": timestamps,
                    "ticker": ["001"] * 3,
                    "sma10": [99.5, 100.0, 100.5],
                }
            ),
        )
        features = {
            f"ohlcv:{name}": {
                "dataset": "ohlcv",
                "name": name,
                "metadata": "ohlcv/metadata.json",
                "availability_delay": "PT0S" if name == "open" else "PT1M",
                "lookahead_safe": True,
            }
            for name in ("open", "high", "low", "close", "volume")
        }
        features["sma:sma10"] = {
            "dataset": "sma",
            "name": "sma10",
            "metadata": "sma/metadata.json",
            "availability_delay": "PT1M",
            "lookahead_safe": True,
        }
        catalog = {
            "catalog_version": 1,
            "name": "test/store",
            "datasets": [
                {
                    "name": family,
                    "kind": "timeseries",
                    "metadata": f"{family}/metadata.json",
                    "time_column": "datetime",
                    "series_keys": ["ticker"],
                    "path_template": f"{family}/year={{year}}/data.parquet",
                    "min_time": "2010-01-04T08:00:00Z",
                    "max_time": "2024-12-30T16:29:00Z",
                }
                for family in ("ohlcv", "sma")
            ]
            + [
                {
                    "name": "symbology",
                    "kind": "table",
                    "metadata": "symbols/metadata.json",
                    "primary_key": ["ticker"],
                    "path_template": "symbols/data.parquet",
                },
                {
                    "name": "markets",
                    "kind": "table",
                    "metadata": "markets/metadata.json",
                    "path_template": "markets/data.parquet",
                },
            ],
            "features": features,
        }
        (cls.fixture_path / "catalog.json").write_text(
            json.dumps(catalog), encoding="utf-8"
        )
        (cls.fixture_path / "symbols").mkdir()
        pq.write_table(
            pa.table(
                {
                    "ticker": ["001"],
                    "company_name": ["Example Company 001"],
                }
            ),
            cls.fixture_path / "symbols" / "data.parquet",
        )
        (cls.fixture_path / "markets").mkdir()
        pq.write_table(
            pa.table(
                {
                    "market_code": ["XSTO", "XNYS"],
                    "weekday": ["Monday", "Monday"],
                }
            ),
            cls.fixture_path / "markets" / "data.parquet",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture_directory.cleanup()

    @classmethod
    def _write_family(cls, family: str, table: pa.Table) -> None:
        partition = cls.fixture_path / family / "year=2024"
        partition.mkdir(parents=True)
        pq.write_table(table, partition / "data.parquet")

    def setUp(self) -> None:
        self.store = FeatureStore(source=self.fixture_path)
        self.store.show_progress = False

    def test_constructor_owns_source_cache_and_token_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = json.loads(
                (self.fixture_path / "catalog.json").read_text(encoding="utf-8")
            )
            catalog["datasets"] = [
                entry for entry in catalog["datasets"] if entry["kind"] == "timeseries"
            ]
            (Path(directory) / "catalog.json").write_text(
                json.dumps(catalog), encoding="utf-8"
            )
            store = FeatureStore(
                source="hf://datasets/owner/features",
                cache=directory,
                token="test-token",
            )

            self.assertEqual(store.source, "hf://datasets/owner/features")
            self.assertEqual(store.cache_path, Path(directory).resolve())
            self.assertEqual(store.token, "test-token")

    def test_relative_source_resolves_from_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_directory = Path(directory) / "store"
            store_directory.mkdir()
            (store_directory / "catalog.json").write_text(
                (self.fixture_path / "catalog.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            for dataset in ("symbols", "markets"):
                destination = store_directory / dataset / "data.parquet"
                destination.parent.mkdir()
                destination.write_bytes(
                    (self.fixture_path / dataset / "data.parquet").read_bytes()
                )
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                store = FeatureStore(source="store")
            finally:
                os.chdir(previous_directory)

        self.assertEqual(store.source, str(store_directory.resolve()))

    def test_connection_configures_duckdb_query_resources(self) -> None:
        store = FeatureStore(source=self.fixture_path)
        connection = store.connection()
        connection.execute("SET memory_limit = '1 GiB'")
        connection.execute("SET threads = 2")

        settings = connection.execute(
            "SELECT current_setting('memory_limit'), current_setting('threads')"
        ).fetchone()
        assert settings is not None
        memory_limit, threads = settings

        self.assertEqual(memory_limit, "1.0 GiB")
        self.assertEqual(threads, 2)

    def test_constructor_can_configure_feature_slice(self) -> None:
        store = FeatureStore(
            source=self.fixture_path,
            features=["ohlcv:close", "sma:sma10"],
            alignment="exact",
            **LOAD_ARGUMENTS,
        )
        store.show_progress = False

        relation = store.connection().table("features")

        self.assertEqual(store.feature_selection, ("ohlcv:close", "sma:sma10"))
        self.assertEqual(store.filters, {"ticker": ("001",)})
        self.assertEqual(relation.columns, ["datetime", "ticker", "close", "sma10"])
        count = relation.aggregate("count(*)").fetchone()
        assert count is not None
        self.assertEqual(count[0], 2)

    def test_filters_preserve_values_and_cover_cached_filters(self) -> None:
        self.assertTrue(
            FeatureStore._covers_filters(
                {"filters": {"ticker": ["001", "002"]}},
                {"ticker": ["001"]},
            )
        )
        self.assertFalse(
            FeatureStore._covers_filters({"filters": {"ticker": ["001"]}}, None)
        )
        store = FeatureStore(
            self.fixture_path,
            features=["ohlcv:close"],
            start=LOAD_ARGUMENTS["start"],
            end=LOAD_ARGUMENTS["end"],
            filters={"ticker": ["1"]},
            alignment="exact",
        )
        self.assertEqual(store.filters, {"ticker": ("1",)})
        with self.assertRaisesRegex(TypeError, "must be a sequence"):
            FeatureStore(
                self.fixture_path,
                features=["ohlcv:close"],
                start=LOAD_ARGUMENTS["start"],
                end=LOAD_ARGUMENTS["end"],
                filters={"ticker": "001"},
                alignment="exact",
            )

    def test_query_stays_lazy_and_selection_is_stable(self) -> None:
        store = FeatureStore(
            source=self.fixture_path,
            features=["ohlcv:close"],
            alignment="exact",
            **LOAD_ARGUMENTS,
        )
        store.show_progress = False

        query = store.query("SELECT avg(close) FROM features")
        plan = query.explain()

        self.assertIn("READ_PARQUET", plan)
        self.assertNotIn("COLUMN_DATA_SCAN", plan)
        average = query.fetchone()
        assert average is not None
        self.assertIsNotNone(average[0])

        volume_store = FeatureStore(
            self.fixture_path,
            features=["ohlcv:volume"],
            alignment="exact",
            **LOAD_ARGUMENTS,
        )

        self.assertEqual(store.feature_selection, ("ohlcv:close",))
        volume = volume_store.query("SELECT sum(volume) FROM features").fetchone()
        assert volume is not None
        self.assertIsNotNone(volume[0])

    def test_features_returns_lazy_narrowed_relation(self) -> None:
        store = FeatureStore(
            source=self.fixture_path,
            features=["ohlcv:close", "ohlcv:volume"],
            alignment="exact",
            **LOAD_ARGUMENTS,
        )

        relation = store.features(
            start="2024-01-02T08:01:00Z",
            end="2024-01-02T08:02:00Z",
            columns=["ticker", "datetime", "close"],
            order_by=["datetime", "ticker"],
        )

        self.assertEqual(relation.columns, ["ticker", "datetime", "close"])
        self.assertEqual(
            relation.fetchall(),
            [("001", datetime(2024, 1, 2, 8, 1, tzinfo=UTC), 101.5)],
        )
        self.assertIn("READ_PARQUET", relation.explain())

    def test_features_requires_configured_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "no configured feature selection"):
            self.store.features()

    def test_feature_batches_yields_bounded_lazy_relations(self) -> None:
        store = FeatureStore(
            source=self.fixture_path,
            features=["ohlcv:close", "ohlcv:volume"],
            alignment="exact",
            **LOAD_ARGUMENTS,
        )

        batches = list(
            store.feature_batches(
                window=timedelta(minutes=1),
                end="2024-01-02T08:01:30Z",
                columns=["datetime", "ticker", "close"],
                order_by=["datetime", "ticker"],
            )
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual(
            [batch.columns for batch in batches],
            [
                ["datetime", "ticker", "close"],
                ["datetime", "ticker", "close"],
            ],
        )
        self.assertEqual(
            [batch.fetchall() for batch in batches],
            [
                [(datetime(2024, 1, 2, 8, 0, tzinfo=UTC), "001", 100.5)],
                [(datetime(2024, 1, 2, 8, 1, tzinfo=UTC), "001", 101.5)],
            ],
        )
        self.assertTrue(all("READ_PARQUET" in batch.explain() for batch in batches))

    def test_feature_batches_validates_window_and_interval(self) -> None:
        store = FeatureStore(
            source=self.fixture_path,
            features=["ohlcv:close"],
            alignment="exact",
            **LOAD_ARGUMENTS,
        )

        with self.assertRaisesRegex(TypeError, "datetime.timedelta"):
            list(store.feature_batches(window=1))
        with self.assertRaisesRegex(ValueError, "window must be positive"):
            list(store.feature_batches(window=timedelta()))
        with self.assertRaisesRegex(ValueError, "end must be later"):
            list(
                store.feature_batches(
                    window=timedelta(days=1),
                    start="2024-01-03T00:00:00Z",
                    end="2024-01-02T00:00:00Z",
                )
            )

    def test_table_returns_registered_table_relation(self) -> None:
        relation = self.store.table("symbology")

        self.assertEqual(relation.columns, ["ticker", "company_name"])
        self.assertEqual(relation.fetchall(), [("001", "Example Company 001")])

    def test_inmemory_materializes_eagerly(self) -> None:
        store = FeatureStore(
            source=self.fixture_path,
            features=["ohlcv:close"],
            alignment="exact",
            inmemory=True,
            **LOAD_ARGUMENTS,
        )
        store.show_progress = False
        plan = store.query("SELECT avg(close) FROM features").explain()

        self.assertIn("SEQ_SCAN", plan)
        self.assertNotIn("READ_PARQUET", plan)

    def configured_store(self, requested_features, **arguments):
        alignment = arguments.pop("alignment", None)
        return FeatureStore(
            self.fixture_path,
            features=requested_features,
            alignment=alignment,
            **arguments,
        )

    def arrow_table(self, requested_features, **arguments):
        alignment = arguments.pop("alignment", "exact")
        store = self.configured_store(
            requested_features, alignment=alignment, **arguments
        )
        return store.connection().table("features").to_arrow_table()

    def test_qualified_unqualified_and_aliased_references(self) -> None:
        qualified = self.arrow_table(["ohlcv:close", "sma:sma10"], **LOAD_ARGUMENTS)
        unqualified = self.arrow_table(["close"], **LOAD_ARGUMENTS)
        aliased = self.arrow_table(
            {"raw_close": "ohlcv:close", "average": "sma:sma10"},
            **LOAD_ARGUMENTS,
        )

        self.assertEqual(
            qualified.column_names, ["datetime", "ticker", "close", "sma10"]
        )
        self.assertEqual(unqualified.column_names, ["datetime", "ticker", "close"])
        self.assertEqual(
            aliased.column_names,
            ["datetime", "ticker", "raw_close", "average"],
        )
        self.assertEqual(qualified.num_rows, 2)

    def test_feature_group_wildcard_selects_every_catalog_feature(self) -> None:
        store = self.configured_store(["ohlcv:*"], alignment="exact", **LOAD_ARGUMENTS)

        self.assertEqual(
            store.feature_selection,
            (
                "ohlcv:open",
                "ohlcv:high",
                "ohlcv:low",
                "ohlcv:close",
                "ohlcv:volume",
            ),
        )
        self.assertEqual(
            store.connection().table("features").columns,
            ["datetime", "ticker", "open", "high", "low", "close", "volume"],
        )
        with self.assertRaisesRegex(ValueError, r"Unknown feature group: missing:\*"):
            self.store._resolve_features(["missing:*"])
        with self.assertRaisesRegex(ValueError, "wildcards cannot have output aliases"):
            self.store._resolve_features({"all_prices": "ohlcv:*"})

    def test_ambiguous_unqualified_reference_lists_choices(self) -> None:
        self.store.catalog()
        self.store._feature_entries["adjusted:close"] = {
            "dataset": "adjusted",
            "name": "close",
            "metadata": "adjusted/metadata.json",
        }

        with self.assertRaisesRegex(
            ValueError, "Ambiguous feature 'close'.*ohlcv:close.*adjusted:close"
        ):
            self.store._resolve_features(["close"])

    def test_output_formats(self) -> None:
        relation = (
            self.configured_store(["ohlcv:close"], alignment="exact", **LOAD_ARGUMENTS)
            .connection()
            .table("features")
        )
        arrow_table = relation.to_arrow_table()
        pandas_frame = relation.df()
        dataset = Dataset(relation.to_arrow_table())

        self.assertIsInstance(arrow_table, pa.Table)
        self.assertIsInstance(pandas_frame, pd.DataFrame)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset.column_names, arrow_table.column_names)

    def test_point_in_time_alignment_respects_availability_delay(self) -> None:
        features = ["ohlcv:open", "ohlcv:close", "sma:sma10"]
        exact = (
            self.configured_store(features, alignment="exact", **LOAD_ARGUMENTS)
            .connection()
            .table("features")
            .order("datetime, ticker")
            .to_arrow_table()
        )
        aligned = (
            self.configured_store(features, alignment="point_in_time", **LOAD_ARGUMENTS)
            .connection()
            .table("features")
            .order("datetime, ticker")
            .to_arrow_table()
        )

        self.assertEqual(aligned.num_rows, exact.num_rows)
        self.assertEqual(aligned["open"].to_pylist(), exact["open"].to_pylist())
        self.assertIsNone(aligned["close"][0].as_py())
        self.assertIsNone(aligned["sma10"][0].as_py())
        self.assertEqual(aligned["close"][1].as_py(), exact["close"][0].as_py())
        self.assertEqual(aligned["sma10"][1].as_py(), exact["sma10"][0].as_py())

    def test_exact_alignment_ignores_availability_delay(self) -> None:
        store = FeatureStore(self.fixture_path, alignment="exact")
        store.show_progress = False
        store._feature_entries["ohlcv:close"]["availability_delay"] = "invalid"

        store._configure_features(["ohlcv:close"], **LOAD_ARGUMENTS)

        relation = store.connection().table("features")
        self.assertEqual(relation.aggregate("count(*)").fetchone(), (2,))

    def test_wholly_out_of_range_query_returns_typed_empty_relation(self) -> None:
        for alignment in ("exact", "point_in_time"):
            for start, end in (
                ("2009-01-01T00:00:00Z", "2009-01-02T00:00:00Z"),
                ("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
            ):
                with self.subTest(alignment=alignment, start=start):
                    relation = (
                        self.configured_store(
                            ["ohlcv:close"],
                            alignment=alignment,
                            start=start,
                            end=end,
                        )
                        .connection()
                        .table("features")
                    )

                    self.assertEqual(relation.columns, ["datetime", "ticker", "close"])
                    self.assertEqual(relation.aggregate("count(*)").fetchone(), (0,))

    def test_point_in_time_lookback_stays_within_family_coverage(self) -> None:
        self.store.catalog()

        paths = self.store._remote_paths(
            "ohlcv",
            datetime(2009, 12, 31, 23, 59, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        )

        self.assertTrue(paths[0].endswith("year=2010/data.parquet"))
        self.assertTrue(paths[-1].endswith("year=2024/data.parquet"))
        self.assertFalse(any("year=2009" in path for path in paths))

        interval = self.store._available_interval(
            "ohlcv",
            datetime(2009, 12, 31, 23, 59, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert interval is not None
        self.assertEqual(interval[0], datetime(2010, 1, 4, 8, 0, tzinfo=UTC))
        self.assertEqual(interval[1], datetime(2024, 12, 30, 16, 29, 0, 1, tzinfo=UTC))

    def test_reads_wildcard_time_series_path_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(self.fixture_path, root, dirs_exist_ok=True)
            catalog_path = root / "catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog["datasets"][0]["path_template"] = "ohlcv/year={year}/*.parquet"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

            store = FeatureStore(
                source=root,
                features=["ohlcv:close"],
                alignment="exact",
                **LOAD_ARGUMENTS,
            )

            self.assertEqual(
                store.query("FROM features").fetchall(),
                [
                    (datetime(2024, 1, 2, 8, 0, tzinfo=UTC), "001", 100.5),
                    (datetime(2024, 1, 2, 8, 1, tzinfo=UTC), "001", 101.5),
                ],
            )

    def test_rejects_naive_query_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include a timezone offset"):
            FeatureStore(
                self.fixture_path,
                features=["ohlcv:close"],
                start="2024-01-02T08:00:00",
                end="2024-01-02T08:02:00Z",
                alignment="exact",
            )

    def test_iso_8601_availability_delay_parser(self) -> None:
        self.assertEqual(
            parse_availability_delay("P1DT2H3M4.5S", "test:value"),
            timedelta(days=1, hours=2, minutes=3, seconds=4.5),
        )
        for invalid in ("1 minute", "P1M", "P1DT", "PT", "P", "-PT1M"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "Invalid availability_delay"):
                    parse_availability_delay(invalid, "test:value")

    def test_point_in_time_alignment_rejects_unsafe_feature(self) -> None:
        store = FeatureStore(self.fixture_path, alignment="point_in_time")
        store._feature_entries["ohlcv:close"]["lookahead_safe"] = False

        with self.assertRaisesRegex(ValueError, "ohlcv:close.*lookahead_safe"):
            store._configure_features(["ohlcv:close"], **LOAD_ARGUMENTS)

    def test_registered_features_are_lazy_and_support_sql(self) -> None:
        store = self.configured_store(
            ["ohlcv:close", "ohlcv:volume"], alignment="exact", **LOAD_ARGUMENTS
        )
        relation = store.connection().table("features")
        result = store.query(
            "SELECT ticker, avg(close) FROM features GROUP BY ticker",
        ).fetchall()
        plan = relation.explain()

        self.assertIsInstance(relation, duckdb.DuckDBPyRelation)
        self.assertIn("READ_PARQUET", plan)
        self.assertNotIn("COLUMN_DATA_SCAN", plan)
        self.assertNotIn("ORDER_BY", plan)
        self.assertEqual(result[0][0], "001")

    def test_inmemory_registers_materialized_features_table(self) -> None:
        store = self.configured_store(
            ["ohlcv:close", "sma:sma10"],
            alignment="point_in_time",
            inmemory=True,
            **LOAD_ARGUMENTS,
        )

        relation = store.connection().table("features")
        plan = relation.explain()

        count = relation.aggregate("count(*)").fetchone()
        assert count is not None
        self.assertEqual(count[0], 2)
        self.assertNotIn("READ_PARQUET", plan)
        self.assertNotIn("ASOF_JOIN", plan)

    def test_registered_table_supports_relations_and_sql(self) -> None:
        relation = (
            self.store.connection()
            .table("symbology")
            .project("ticker, company_name")
            .filter("ticker = '001'")
        )

        self.assertEqual(relation.columns, ["ticker", "company_name"])
        self.assertEqual(relation.fetchall(), [("001", "Example Company 001")])
        self.assertIn("READ_PARQUET", relation.explain())
        result = self.store.query(
            "SELECT company_name FROM symbology WHERE ticker = '001'"
        ).fetchone()
        self.assertEqual(result, ("Example Company 001",))

    def test_feature_query_joins_registered_table(self) -> None:
        store = self.configured_store(
            ["ohlcv:close"], alignment="exact", **LOAD_ARGUMENTS
        )

        rows = store.query(
            "SELECT features.ticker, symbology.company_name "
            "FROM features JOIN symbology USING (ticker)"
        ).fetchall()

        self.assertEqual(rows, [("001", "Example Company 001")] * 2)

    def test_all_catalog_tables_are_available(self) -> None:
        self.assertEqual(
            self.store.connection().table("symbology").aggregate("count(*)").fetchone(),
            (1,),
        )
        self.assertEqual(
            self.store.connection().table("markets").aggregate("count(*)").fetchone(),
            (2,),
        )

    def test_remote_table_is_cached_as_complete_offline_snapshot(self) -> None:
        class FakeFilesystem:
            def __init__(self, files: dict[str, bytes]) -> None:
                self.files = files
                self.opens: list[str] = []

            def open(self, path: str, mode: str):
                self.opens.append(path)
                content = self.files[path]
                if "b" in mode:
                    return io.BytesIO(content)
                return io.StringIO(content.decode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            catalog = json.loads(
                (self.fixture_path / "catalog.json").read_text(encoding="utf-8")
            )
            symbology = next(
                entry for entry in catalog["datasets"] if entry["name"] == "symbology"
            )
            symbology.pop("path_template")
            catalog_path = cache.parent / f"{cache.name}-catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            metadata = {
                "dataset": "symbology",
                "kind": "table",
                "storage": {
                    "format": "parquet",
                    "path_template": "symbols/data.parquet",
                },
            }
            remote_parquet = (
                self.fixture_path / "symbols" / "data.parquet"
            ).read_bytes()
            remote_markets = (
                self.fixture_path / "markets" / "data.parquet"
            ).read_bytes()
            filesystem = FakeFilesystem(
                {
                    "hf://datasets/owner/store/symbols/metadata.json": json.dumps(
                        metadata
                    ).encode("utf-8"),
                    "hf://datasets/owner/store/symbols/data.parquet": remote_parquet,
                    "hf://datasets/owner/store/markets/data.parquet": remote_markets,
                }
            )
            with patch.object(
                FeatureStore, "_huggingface_filesystem", lambda _: filesystem
            ):
                store = FeatureStore(
                    "hf://datasets/owner/store",
                    cache=cache,
                    catalog_path=catalog_path,
                )
            store.show_progress = False

            rows = store.query("SELECT * FROM symbology").fetchall()
            repeated_rows = store.query("SELECT * FROM symbology").fetchall()
            manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
            offline_store = FeatureStore(cache)
            offline_rows = offline_store.query("SELECT * FROM symbology").fetchall()
            offline_markets = offline_store.query("SELECT * FROM markets").fetchall()

            catalog_path.unlink()

        self.assertEqual(rows, [("001", "Example Company 001")])
        self.assertEqual(repeated_rows, rows)
        self.assertEqual(offline_rows, rows)
        self.assertEqual(len(offline_markets), 2)
        self.assertEqual(
            filesystem.opens.count("hf://datasets/owner/store/symbols/data.parquet"),
            1,
        )
        self.assertEqual(
            manifest["fragments"][-2:],
            [
                {
                    "kind": "table",
                    "dataset": "symbology",
                    "path": "symbols/data.parquet",
                },
                {
                    "kind": "table",
                    "dataset": "markets",
                    "path": "markets/data.parquet",
                },
            ],
        )

    def test_table_only_catalog_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "items").mkdir()
            pq.write_table(
                pa.table({"value": ["one", "two"]}),
                root / "items" / "data.parquet",
            )
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "catalog_version": 1,
                        "datasets": [
                            {
                                "name": "items",
                                "kind": "table",
                                "metadata": "items/metadata.json",
                                "path_template": "items/data.parquet",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = FeatureStore(root).query("SELECT * FROM items").fetchall()

        self.assertEqual(rows, [("one",), ("two",)])

    def test_timeseries_supports_arbitrary_time_and_composite_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timestamps = pa.array(
                [datetime(2024, 1, 1, hour, tzinfo=UTC) for hour in (0, 1)],
                type=pa.timestamp("us", tz="UTC"),
            )
            for name, feature, values in (
                ("temperature", "celsius", [1.5, 2.0]),
                ("humidity", "percent", [80.0, 75.0]),
            ):
                partition = root / name / "year=2024"
                partition.mkdir(parents=True)
                pq.write_table(
                    pa.table(
                        {
                            "observed_at": timestamps,
                            "station_id": ["north", "north"],
                            "sensor_id": ["outdoor", "outdoor"],
                            feature: values,
                        }
                    ),
                    partition / "data.parquet",
                )
            datasets = [
                {
                    "name": name,
                    "kind": "timeseries",
                    "metadata": f"{name}/metadata.json",
                    "time_column": "observed_at",
                    "series_keys": ["station_id", "sensor_id"],
                    "path_template": f"{name}/year={{year}}/data.parquet",
                    "min_time": "2024-01-01T00:00:00Z",
                    "max_time": "2024-01-01T01:00:00Z",
                }
                for name in ("temperature", "humidity")
            ]
            features = {
                "temperature:celsius": {
                    "dataset": "temperature",
                    "name": "celsius",
                    "availability_delay": "PT0S",
                    "lookahead_safe": True,
                },
                "humidity:percent": {
                    "dataset": "humidity",
                    "name": "percent",
                    "availability_delay": "PT0S",
                    "lookahead_safe": True,
                },
            }
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "catalog_version": 1,
                        "datasets": datasets,
                        "features": features,
                    }
                ),
                encoding="utf-8",
            )
            store = FeatureStore(
                root,
                features=["temperature:celsius", "humidity:percent"],
                start="2024-01-01T00:00:00Z",
                end="2024-01-01T02:00:00Z",
                alignment="exact",
            )
            store.show_progress = False

            table = store.connection().table("features").to_arrow_table()

        self.assertEqual(
            table.column_names,
            ["observed_at", "station_id", "sensor_id", "celsius", "percent"],
        )
        self.assertEqual(table.num_rows, 2)

    def test_constructor_requires_alignment_and_sql_supports_subslices(self) -> None:
        with self.assertRaisesRegex(TypeError, "alignment is required"):
            self.configured_store(
                ["ohlcv:close"],
                start="2024-01-02T08:00:00Z",
                end="2024-01-02T08:03:00Z",
                filters={"ticker": ["001"]},
            )
        store = self.configured_store(
            ["ohlcv:close"],
            alignment="exact",
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:03:00Z",
            filters={"ticker": ["001"]},
        )

        relation = store.query(
            "SELECT * FROM features "
            "WHERE datetime >= TIMESTAMPTZ '2024-01-02T08:01:00Z' "
            "AND datetime < TIMESTAMPTZ '2024-01-02T08:03:00Z'"
        )
        count = relation.aggregate("count(*)").fetchone()
        assert count is not None
        self.assertEqual(count[0], 2)

    def test_native_arrow_batch_reader(self) -> None:
        store = self.configured_store(
            ["ohlcv:close"], alignment="exact", **LOAD_ARGUMENTS
        )

        batches = list(
            store.connection().table("features").to_arrow_reader(batch_size=1)
        )

        self.assertEqual([batch.num_rows for batch in batches], [1, 1])

    def test_unsupported_catalog_versions_are_rejected(self) -> None:
        for version in (2, 3):
            with (
                self.subTest(version=version),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "catalog.json").write_text(
                    json.dumps({"catalog_version": version}), encoding="utf-8"
                )

                with self.assertRaisesRegex(
                    ValueError, f"Unsupported catalog version: {version}"
                ):
                    FeatureStore(root)


if __name__ == "__main__":
    unittest.main()
