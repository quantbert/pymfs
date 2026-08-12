import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_catalog import build_catalog, inspect_partition
from gendata import (
    generate_markets,
    generate_symbology,
    session_minutes,
    write_table_dataset,
)
from hfupload import parse_destination


class UploadConfigurationTests(unittest.TestCase):
    def test_parses_supported_destinations(self) -> None:
        self.assertEqual(
            parse_destination("hf://buckets/owner/store"),
            ("buckets", "owner/store"),
        )
        self.assertEqual(
            parse_destination("hf://datasets/owner/store/"),
            ("datasets", "owner/store"),
        )

    def test_rejects_invalid_destination(self) -> None:
        with self.assertRaisesRegex(ValueError, "destination must be"):
            parse_destination("owner/store")


class CatalogTests(unittest.TestCase):
    def test_uses_configured_store_name_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            for dataset in ("ohlcv", "sma"):
                partition = data_root / dataset / "year=2024"
                partition.mkdir(parents=True)
                pq.write_table(
                    pa.table(
                        {
                            "datetime": pa.array(
                                [datetime(2024, 1, 2, 8, 0, tzinfo=UTC)],
                                type=pa.timestamp("us", tz="UTC"),
                            )
                        }
                    ),
                    partition / "data.parquet",
                )
            write_table_dataset(
                data_root / "symbols" / "data.parquet",
                generate_symbology(["007"]),
                overwrite=False,
            )
            write_table_dataset(
                data_root / "markets" / "data.parquet",
                generate_markets(),
                overwrite=False,
            )

            catalog = build_catalog(
                data_root,
                store_name="owner/store",
                source="hf://buckets/owner/store",
            )
            readme = (data_root / "README.md").read_text(encoding="utf-8")

        self.assertEqual(catalog["name"], "owner/store")
        self.assertEqual(catalog["catalog_version"], 3)
        self.assertEqual(
            [dataset["name"] for dataset in catalog["datasets"]],
            ["ohlcv", "sma", "symbology", "markets"],
        )
        self.assertEqual(catalog["datasets"][0]["min_time"], "2024-01-02T08:00:00Z")
        self.assertEqual(catalog["datasets"][2]["primary_key"], ["ticker"])
        self.assertNotIn("primary_key", catalog["datasets"][3])
        self.assertIn('source="hf://buckets/owner/store"', readme)

    def test_rejects_non_utc_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pq.write_table(
                pa.table(
                    {
                        "datetime": pa.array(
                            [datetime.fromisoformat("2024-01-02T09:00:00+01:00")]
                        )
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "timezone-aware UTC"):
                inspect_partition(path, Path(directory), "datetime")


class GeneratedDatasetTests(unittest.TestCase):
    def test_session_timestamps_are_normalized_to_utc(self) -> None:
        winter, summer = session_minutes([date(2024, 1, 2), date(2024, 7, 1)])[::510]

        self.assertEqual(winter, datetime(2024, 1, 2, 8, 0, tzinfo=UTC))
        self.assertEqual(summer, datetime(2024, 7, 1, 7, 0, tzinfo=UTC))

    def test_generates_one_keyed_symbology_row_per_ticker(self) -> None:
        table = generate_symbology(["007", "042"])

        self.assertEqual(table.schema.field("ticker").nullable, False)
        self.assertEqual(table.column("ticker").to_pylist(), ["007", "042"])
        self.assertEqual(table.column("isin").to_pylist(), ["SE0000000007", "SE0000000042"])
        self.assertEqual(table.column("cik").to_pylist(), ["0000000008", "0000000043"])

    def test_generates_non_keyed_market_hours_rows(self) -> None:
        table = generate_markets()

        self.assertEqual(table.num_rows, 10)
        self.assertEqual(set(table.column("market_code").to_pylist()), {"XSTO", "XNYS"})
        self.assertNotIn("id", table.column_names)

    def test_writes_single_file_table_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "symbols" / "data.parquet"
            write_table_dataset(destination, generate_symbology(["007"]), overwrite=False)

            table = pq.read_table(destination)

        self.assertEqual(table.column("ticker").to_pylist(), ["007"])


if __name__ == "__main__":
    unittest.main()