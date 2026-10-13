"""Stream and validate the Retailrocket CSVs into offline-only processed data."""

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVENT_FILES = ("events.csv",)
PROPERTY_FILES = ("item_properties_part1.csv", "item_properties_part2.csv")
CATEGORY_FILE = "category_tree.csv"
OUTPUT_FILES = (
	"interactions.csv",
	"category_tree.csv",
	"item_properties.csv",
	"validation_report.json",
)
EVENT_MAP = {"view": "view", "addtocart": "cart", "transaction": "purchase"}
EVENT_COLUMNS = ("timestamp", "visitorid", "event", "itemid", "transactionid")
CATEGORY_COLUMNS = ("categoryid", "parentid")
PROPERTY_COLUMNS = ("timestamp", "itemid", "property", "value")


def _digest(values):
	encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def _integer(value, allow_empty=False):
	if value is None:
		value = ""
	elif not isinstance(value, str):
		value = str(value)
	value = value.strip()
	if not value and allow_empty:
		return None
	if not value:
		raise ValueError("missing_numeric_value")
	try:
		parsed = int(value)
	except (TypeError, ValueError) as error:
		raise ValueError("invalid_integer") from error
	if parsed < 0:
		raise ValueError("negative_integer")
	return parsed


def _timestamp(value):
	timestamp_ms = _integer(value)
	try:
		parsed = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
	except (OverflowError, OSError, ValueError) as error:
		raise ValueError("invalid_timestamp") from error
	return timestamp_ms, parsed.isoformat()


def _check_header(reader, required, path):
	try:
		header = next(reader)
	except StopIteration as error:
		raise ValueError("{} is empty.".format(path)) from error
	if len(header) != len(set(header)):
		raise ValueError("{} contains duplicate column names.".format(path))
	missing = sorted(set(required) - set(header))
	if missing:
		raise ValueError("{} is missing required columns: {}".format(path, ", ".join(missing)))
	return header


def _new_file_stats(columns):
	return {
		"source_rows": 0,
		"output_rows": 0,
		"duplicate_rows": 0,
		"rejected_rows": 0,
		"missing_values": dict.fromkeys(columns, 0),
		"errors": {},
	}


def _record_missing(stats, columns, row):
	for column in columns:
		value = row.get(column)
		if value is None or not str(value).strip():
			stats["missing_values"][column] += 1


def _reject(stats, errors, reason):
	stats["rejected_rows"] += 1
	errors[reason] += 1


def _file_reader(path, required):
	handle = path.open("r", encoding="utf-8-sig", newline="")
	reader = csv.reader(handle)
	try:
		header = _check_header(reader, required, path)
	except Exception:
		handle.close()
		raise
	return handle, reader, header


def _write_csv(path, header):
	handle = path.open("w", encoding="utf-8", newline="")
	writer = csv.writer(handle)
	writer.writerow(header)
	return handle, writer, header


def _publish_outputs(temporary_dir, output_dir):
	created = []
	try:
		for filename in OUTPUT_FILES:
			temporary_path = Path(temporary_dir) / filename
			destination = output_dir / filename
			os.link(str(temporary_path), str(destination))
			created.append(destination)
	except OSError:
		for destination in created:
			destination.unlink()
		raise


def preprocess_retailrocket(raw_dir=DEFAULT_RAW_DIR, processed_dir=DEFAULT_PROCESSED_DIR):
	"""Create normalized CSVs and a validation report without loading a source file fully."""
	raw_dir = Path(raw_dir)
	processed_dir = Path(processed_dir)
	input_names = EVENT_FILES + (CATEGORY_FILE,) + PROPERTY_FILES
	missing_inputs = [name for name in input_names if not (raw_dir / name).is_file()]
	if missing_inputs:
		raise FileNotFoundError("Missing input files: {}".format(", ".join(missing_inputs)))

	processed_dir.mkdir(parents=True, exist_ok=True)
	existing_outputs = [name for name in OUTPUT_FILES if (processed_dir / name).exists()]
	if existing_outputs:
		raise FileExistsError(
			"Refusing to overwrite existing output(s): {}".format(", ".join(existing_outputs))
		)

	report = {
		"dataset": "Retailrocket ecommerce events",
		"processor_version": 1,
		"processed_at_utc": datetime.now(timezone.utc).isoformat(),
		"id_namespace": "Retailrocket visitorid/itemid; not DealoRa database IDs",
		"timestamp_format": "UTC ISO-8601 derived from source epoch milliseconds",
		"event_mapping": EVENT_MAP,
		"schema": {
			"events.csv": {
				"timestamp": "non-negative integer epoch milliseconds",
				"visitorid": "non-negative integer visitor ID",
				"event": "string event label",
				"itemid": "non-negative integer item ID",
				"transactionid": "nullable non-negative integer; required for transaction events",
			},
			"category_tree.csv": {"categoryid": "integer", "parentid": "nullable integer"},
			"item_properties_part*.csv": {
				"timestamp": "non-negative integer epoch milliseconds",
				"itemid": "non-negative integer item ID",
				"property": "string property key",
				"value": "opaque string; preserved as provided",
			},
		},
		"files": {},
		"event_types": {},
		"mapped_event_types": {},
		"unique_visitors": 0,
		"unique_items_in_events": 0,
		"unique_items_in_properties": 0,
		"unique_items_all_sources": 0,
		"processing_errors": {},
	}

	error_counts = Counter()
	event_type_counts = Counter()
	mapped_event_counts = Counter()
	with tempfile.TemporaryDirectory(prefix=".retailrocket-", dir=str(processed_dir)) as temporary_dir:
		temporary_dir = Path(temporary_dir)
		database = sqlite3.connect(str(temporary_dir / "dedupe.sqlite3"))
		database.execute("PRAGMA journal_mode=OFF")
		database.execute("PRAGMA synchronous=OFF")
		database.execute("PRAGMA cache_size=-65536")
		database.execute("PRAGMA temp_store=MEMORY")
		database.execute("CREATE TABLE seen_events (digest TEXT PRIMARY KEY) WITHOUT ROWID")
		database.execute("CREATE TABLE seen_properties (digest TEXT PRIMARY KEY) WITHOUT ROWID")
		database.execute("CREATE TABLE seen_categories (category_id INTEGER PRIMARY KEY) WITHOUT ROWID")
		database.execute("CREATE TABLE visitors (visitor_id INTEGER PRIMARY KEY) WITHOUT ROWID")
		database.execute("CREATE TABLE event_items (item_id INTEGER PRIMARY KEY) WITHOUT ROWID")
		database.execute("CREATE TABLE property_items (item_id INTEGER PRIMARY KEY) WITHOUT ROWID")

		interactions_handle = None
		category_handle = None
		properties_handle = None
		try:
			interactions_handle, interactions_writer, _ = _write_csv(
				temporary_dir / "interactions.csv",
				("visitor_id", "item_id", "event_type", "timestamp_ms", "timestamp_utc", "transaction_id"),
			)
			category_handle, category_writer, _ = _write_csv(
				temporary_dir / "category_tree.csv", ("category_id", "parent_id")
			)
			properties_handle, properties_writer, _ = _write_csv(
				temporary_dir / "item_properties.csv", ("timestamp_ms", "item_id", "property", "value")
			)
			for filename in EVENT_FILES:
				path = raw_dir / filename
				stats = _new_file_stats(EVENT_COLUMNS)
				errors = Counter()
				handle, reader, header = _file_reader(path, EVENT_COLUMNS)
				try:
					positions = {name: header.index(name) for name in EVENT_COLUMNS}
					for values in reader:
						stats["source_rows"] += 1
						if len(values) != len(header):
							_record_missing(stats, EVENT_COLUMNS, dict(zip(header, values)))
							_reject(stats, errors, "malformed_column_count")
							continue
						row = {name: values[index] for name, index in positions.items()}
						_record_missing(stats, EVENT_COLUMNS, row)
						try:
							timestamp_ms, timestamp_utc = _timestamp(row["timestamp"])
							visitor_id = _integer(row["visitorid"])
							item_id = _integer(row["itemid"])
						except ValueError as error:
							_reject(stats, errors, str(error) if str(error) in {"negative_integer", "invalid_timestamp"} else "invalid_numeric_field")
							continue
						event_name = (row["event"] or "").strip().lower()
						event_type_counts[event_name or "<missing>"] += 1
						activity_type = EVENT_MAP.get(event_name)
						if activity_type is None:
							_reject(stats, errors, "unsupported_event_type")
							continue
						transaction_id = (row["transactionid"] or "").strip() or None
						if event_name == "transaction":
							try:
								transaction_id = str(_integer(transaction_id))
							except (TypeError, ValueError):
								_reject(stats, errors, "transaction_missing_or_invalid_id")
								continue
						key = [timestamp_ms, visitor_id, event_name, item_id, transaction_id]
						inserted = database.execute(
							"INSERT OR IGNORE INTO seen_events(digest) VALUES (?)", (_digest(key),)
						).rowcount
						if not inserted:
							stats["duplicate_rows"] += 1
							continue
						interactions_writer.writerow((
							visitor_id, item_id, activity_type, timestamp_ms, timestamp_utc, transaction_id or "",
						))
						database.execute("INSERT OR IGNORE INTO visitors(visitor_id) VALUES (?)", (visitor_id,))
						database.execute("INSERT OR IGNORE INTO event_items(item_id) VALUES (?)", (item_id,))
						stats["output_rows"] += 1
						mapped_event_counts[activity_type] += 1
				finally:
					handle.close()
				database.commit()
				stats["errors"] = dict(sorted(errors.items()))
				report["files"][filename] = stats
				error_counts.update(errors)

			path = raw_dir / CATEGORY_FILE
			stats = _new_file_stats(CATEGORY_COLUMNS)
			errors = Counter()
			handle, reader, header = _file_reader(path, CATEGORY_COLUMNS)
			try:
				positions = {name: header.index(name) for name in CATEGORY_COLUMNS}
				for values in reader:
					stats["source_rows"] += 1
					if len(values) != len(header):
						_record_missing(stats, CATEGORY_COLUMNS, dict(zip(header, values)))
						_reject(stats, errors, "malformed_column_count")
						continue
					row = {name: values[index] for name, index in positions.items()}
					_record_missing(stats, CATEGORY_COLUMNS, row)
					try:
						category_id = _integer(row["categoryid"])
						parent_id = _integer(row["parentid"], allow_empty=True)
					except ValueError:
						_reject(stats, errors, "invalid_category_id")
						continue
					if category_id == parent_id:
						_reject(stats, errors, "self_referencing_category")
						continue
					inserted = database.execute(
						"INSERT OR IGNORE INTO seen_categories(category_id) VALUES (?)", (category_id,)
					).rowcount
					if not inserted:
						stats["duplicate_rows"] += 1
						continue
					category_writer.writerow((category_id, "" if parent_id is None else parent_id))
					stats["output_rows"] += 1
			finally:
				handle.close()
			database.commit()
			stats["errors"] = dict(sorted(errors.items()))
			report["files"][CATEGORY_FILE] = stats
			error_counts.update(errors)

			for filename in PROPERTY_FILES:
				path = raw_dir / filename
				stats = _new_file_stats(PROPERTY_COLUMNS)
				errors = Counter()
				handle, reader, header = _file_reader(path, PROPERTY_COLUMNS)
				try:
					positions = {name: header.index(name) for name in PROPERTY_COLUMNS}
					for values in reader:
						stats["source_rows"] += 1
						if len(values) != len(header):
							_record_missing(stats, PROPERTY_COLUMNS, dict(zip(header, values)))
							_reject(stats, errors, "malformed_column_count")
							continue
						row = {name: values[index] for name, index in positions.items()}
						_record_missing(stats, PROPERTY_COLUMNS, row)
						property_name = (row["property"] or "").strip()
						if not property_name:
							_reject(stats, errors, "missing_property_name")
							continue
						try:
							timestamp_ms, _timestamp_utc = _timestamp(row["timestamp"])
							item_id = _integer(row["itemid"])
						except ValueError as error:
							reason = str(error)
							if reason not in {"negative_integer", "invalid_timestamp"}:
								reason = "invalid_numeric_field"
							_reject(stats, errors, reason)
							continue
						value = row["value"] or ""
						key = [timestamp_ms, item_id, property_name, value]
						inserted = database.execute(
							"INSERT OR IGNORE INTO seen_properties(digest) VALUES (?)", (_digest(key),)
						).rowcount
						if not inserted:
							stats["duplicate_rows"] += 1
							continue
						properties_writer.writerow((timestamp_ms, item_id, property_name, value))
						database.execute("INSERT OR IGNORE INTO property_items(item_id) VALUES (?)", (item_id,))
						stats["output_rows"] += 1
						if stats["source_rows"] % 100000 == 0:
							database.commit()
				finally:
					handle.close()
				database.commit()
				stats["errors"] = dict(sorted(errors.items()))
				report["files"][filename] = stats
				error_counts.update(errors)

			interactions_handle.close()
			category_handle.close()
			properties_handle.close()
			report["event_types"] = dict(sorted(event_type_counts.items()))
			report["mapped_event_types"] = dict(sorted(mapped_event_counts.items()))
			report["unique_visitors"] = database.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
			report["unique_items_in_events"] = database.execute("SELECT COUNT(*) FROM event_items").fetchone()[0]
			report["unique_items_in_properties"] = database.execute("SELECT COUNT(*) FROM property_items").fetchone()[0]
			database.execute(
				"INSERT OR IGNORE INTO event_items SELECT item_id FROM property_items"
			)
			report["unique_items_all_sources"] = database.execute("SELECT COUNT(*) FROM event_items").fetchone()[0]
			report["processing_errors"] = dict(sorted(error_counts.items()))

			report_path = temporary_dir / "validation_report.json"
			with report_path.open("w", encoding="utf-8") as handle:
				json.dump(report, handle, indent=2, sort_keys=True)
				handle.write("\n")
			_publish_outputs(temporary_dir, processed_dir)
		finally:
			for handle in (interactions_handle, category_handle, properties_handle):
				if handle is not None:
					handle.close()
			database.close()
	return report


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
	parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
	arguments = parser.parse_args()
	try:
		report = preprocess_retailrocket(arguments.raw_dir, arguments.processed_dir)
	except (OSError, ValueError, sqlite3.Error, csv.Error) as error:
		print("Retailrocket preprocessing failed: {}".format(error), file=sys.stderr)
		return 1
	print(json.dumps({
		"processed_dir": str(arguments.processed_dir),
		"rows": {name: stats["output_rows"] for name, stats in report["files"].items()},
		"unique_visitors": report["unique_visitors"],
		"unique_items_all_sources": report["unique_items_all_sources"],
		"event_types": report["event_types"],
		"processing_errors": report["processing_errors"],
	}, indent=2))
	return 0


if __name__ == "__main__":
	sys.exit(main())