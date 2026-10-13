"""Stream Retailrocket raw CSVs into EDA and data-quality JSON. Never writes to data/raw."""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_retailrocket import (  # noqa: E402
	CATEGORY_COLUMNS,
	CATEGORY_FILE,
	EVENT_COLUMNS,
	EVENT_FILES,
	EVENT_MAP,
	PROPERTY_COLUMNS,
	PROPERTY_FILES,
	_check_header,
	_integer,
	_timestamp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUALITY_FILE = "data_quality_report.json"
EDA_FILE = "eda_summary.json"
PROPERTY_SAMPLE_ROWS = 250000


def _refuse_existing(output_dir, names):
	existing = [name for name in names if (output_dir / name).exists()]
	if existing:
		raise FileExistsError("Refusing to overwrite existing output(s): {}".format(", ".join(existing)))


def _stream_events(path, daily, event_types, mapped, item_counts, visitor_counts, quality):
	handle = path.open("r", encoding="utf-8-sig", newline="")
	reader = csv.reader(handle)
	try:
		header = _check_header(reader, EVENT_COLUMNS, path)
		positions = {name: header.index(name) for name in EVENT_COLUMNS}
		for values in reader:
			quality["events_source_rows"] += 1
			if len(values) != len(header):
				quality["events_rejected"]["malformed_column_count"] += 1
				continue
			row = {name: values[index] for name, index in positions.items()}
			try:
				timestamp_ms, timestamp_utc = _timestamp(row["timestamp"])
				visitor_id = _integer(row["visitorid"])
				item_id = _integer(row["itemid"])
			except ValueError as error:
				reason = str(error) if str(error) in {"negative_integer", "invalid_timestamp"} else "invalid_numeric_field"
				quality["events_rejected"][reason] += 1
				continue
			event_name = (row["event"] or "").strip().lower()
			event_types[event_name or "<missing>"] += 1
			activity_type = EVENT_MAP.get(event_name)
			if activity_type is None:
				quality["events_rejected"]["unsupported_event_type"] += 1
				continue
			transaction_id = (row["transactionid"] or "").strip() or None
			if event_name == "transaction":
				try:
					_integer(transaction_id)
				except (TypeError, ValueError):
					quality["events_rejected"]["transaction_missing_or_invalid_id"] += 1
					continue
			quality["events_kept"] += 1
			mapped[activity_type] += 1
			day = timestamp_utc[:10]
			daily[day][activity_type] += 1
			item_counts[item_id] += 1
			visitor_counts[visitor_id] += 1
			if quality["events_kept"] == 1:
				quality["min_timestamp_utc"] = timestamp_utc
			quality["max_timestamp_utc"] = timestamp_utc
			if timestamp_utc < quality["min_timestamp_utc"]:
				quality["min_timestamp_utc"] = timestamp_utc
	finally:
		handle.close()


def _stream_categories(path, quality):
	handle = path.open("r", encoding="utf-8-sig", newline="")
	reader = csv.reader(handle)
	try:
		header = _check_header(reader, CATEGORY_COLUMNS, path)
		positions = {name: header.index(name) for name in CATEGORY_COLUMNS}
		seen = set()
		for values in reader:
			quality["category_source_rows"] += 1
			if len(values) != len(header):
				quality["category_rejected"]["malformed_column_count"] += 1
				continue
			row = {name: values[index] for name, index in positions.items()}
			try:
				category_id = _integer(row["categoryid"])
				parent_id = _integer(row["parentid"], allow_empty=True)
			except ValueError:
				quality["category_rejected"]["invalid_category_id"] += 1
				continue
			if category_id == parent_id:
				quality["category_rejected"]["self_referencing_category"] += 1
				continue
			if category_id in seen:
				quality["category_duplicate_ids"] += 1
				continue
			seen.add(category_id)
			quality["category_kept"] += 1
			if parent_id is None:
				quality["category_root_nodes"] += 1
	finally:
		handle.close()


def _sample_properties(path, quality, sample_limit):
	handle = path.open("r", encoding="utf-8-sig", newline="")
	reader = csv.reader(handle)
	file_stats = {
		"source_rows_sampled": 0,
		"kept": 0,
		"rejected": Counter(),
		"missing_value": 0,
		"distinct_properties_sampled": 0,
	}
	properties = Counter()
	try:
		header = _check_header(reader, PROPERTY_COLUMNS, path)
		positions = {name: header.index(name) for name in PROPERTY_COLUMNS}
		for values in reader:
			if file_stats["source_rows_sampled"] >= sample_limit:
				file_stats["stopped_early"] = True
				break
			file_stats["source_rows_sampled"] += 1
			if len(values) != len(header):
				file_stats["rejected"]["malformed_column_count"] += 1
				continue
			row = {name: values[index] for name, index in positions.items()}
			property_name = (row["property"] or "").strip()
			if not property_name:
				file_stats["rejected"]["missing_property_name"] += 1
				continue
			try:
				_timestamp(row["timestamp"])
				_integer(row["itemid"])
			except ValueError as error:
				reason = str(error) if str(error) in {"negative_integer", "invalid_timestamp"} else "invalid_numeric_field"
				file_stats["rejected"][reason] += 1
				continue
			if not (row.get("value") or "").strip():
				file_stats["missing_value"] += 1
			properties[property_name] += 1
			file_stats["kept"] += 1
	finally:
		handle.close()
	file_stats["rejected"] = dict(sorted(file_stats["rejected"].items()))
	file_stats["distinct_properties_sampled"] = len(properties)
	file_stats["top_properties_sampled"] = properties.most_common(8)
	quality["property_samples"][path.name] = file_stats


def _activity_histogram(visitor_counts):
	buckets = Counter()
	for count in visitor_counts.values():
		if count == 1:
			buckets["1"] += 1
		elif count <= 5:
			buckets["2-5"] += 1
		elif count <= 20:
			buckets["6-20"] += 1
		elif count <= 50:
			buckets["21-50"] += 1
		else:
			buckets["51+"] += 1
	return dict(sorted(buckets.items(), key=lambda item: item[0]))


def build_eda(raw_dir=DEFAULT_RAW_DIR, processed_dir=DEFAULT_PROCESSED_DIR, property_sample_rows=PROPERTY_SAMPLE_ROWS):
	raw_dir = Path(raw_dir)
	processed_dir = Path(processed_dir)
	processed_dir.mkdir(parents=True, exist_ok=True)
	_refuse_existing(processed_dir, (QUALITY_FILE, EDA_FILE))
	missing = [name for name in EVENT_FILES + (CATEGORY_FILE,) if not (raw_dir / name).is_file()]
	if missing:
		raise FileNotFoundError("Missing input files: {}".format(", ".join(missing)))

	quality = {
		"dataset": "Retailrocket ecommerce events",
		"processor": "eda_retailrocket.py",
		"processed_at_utc": datetime.now(timezone.utc).isoformat(),
		"raw_dir": str(raw_dir),
		"raw_files_never_modified": True,
		"id_namespace": "Retailrocket visitorid/itemid; not DealoRa database IDs",
		"events_source_rows": 0,
		"events_kept": 0,
		"events_rejected": Counter(),
		"min_timestamp_utc": None,
		"max_timestamp_utc": None,
		"category_source_rows": 0,
		"category_kept": 0,
		"category_duplicate_ids": 0,
		"category_root_nodes": 0,
		"category_rejected": Counter(),
		"property_sample_row_cap_per_file": property_sample_rows,
		"property_full_files_not_loaded": True,
		"property_samples": {},
		"notes": [
			"Events are validated with the same rules as process_retailrocket.py.",
			"Item property files are sampled only; hashed Retailrocket property values are opaque.",
			"Incomplete .retailrocket-* directories under data/processed are left untouched.",
		],
	}
	event_types = Counter()
	mapped = Counter()
	item_counts = Counter()
	visitor_counts = Counter()

	class DefaultDaily(dict):
		def __missing__(self, key):
			value = Counter()
			self[key] = value
			return value

	daily = DefaultDaily()
	for filename in EVENT_FILES:
		_stream_events(raw_dir / filename, daily, event_types, mapped, item_counts, visitor_counts, quality)
	_stream_categories(raw_dir / CATEGORY_FILE, quality)
	for filename in PROPERTY_FILES:
		path = raw_dir / filename
		if path.is_file():
			_sample_properties(path, quality, property_sample_rows)

	quality["events_rejected"] = dict(sorted(quality["events_rejected"].items()))
	quality["category_rejected"] = dict(sorted(quality["category_rejected"].items()))
	quality["events_removed"] = quality["events_source_rows"] - quality["events_kept"]
	quality["retention_rate"] = (
		quality["events_kept"] / float(quality["events_source_rows"])
		if quality["events_source_rows"] else None
	)

	daily_series = []
	for day in sorted(daily):
		row = {"date": day, "view": 0, "cart": 0, "purchase": 0}
		row.update(dict(daily[day]))
		row["total"] = row.get("view", 0) + row.get("cart", 0) + row.get("purchase", 0)
		daily_series.append(row)

	eda = {
		"dataset": "Retailrocket ecommerce events",
		"generated_at_utc": quality["processed_at_utc"],
		"id_namespace": quality["id_namespace"],
		"records_before_cleaning": quality["events_source_rows"],
		"records_after_cleaning": quality["events_kept"],
		"records_removed": quality["events_removed"],
		"unique_visitors": len(visitor_counts),
		"unique_items": len(item_counts),
		"event_types_raw": dict(sorted(event_types.items())),
		"mapped_event_types": dict(sorted(mapped.items())),
		"daily_events": daily_series,
		"top_items": [
			{"item_id": item_id, "interactions": count}
			for item_id, count in item_counts.most_common(20)
		],
		"visitor_activity_histogram": _activity_histogram(visitor_counts),
		"date_range_utc": {
			"min": quality["min_timestamp_utc"],
			"max": quality["max_timestamp_utc"],
		},
	}

	with (processed_dir / QUALITY_FILE).open("w", encoding="utf-8") as handle:
		json.dump(quality, handle, indent=2, sort_keys=True)
		handle.write("\n")
	with (processed_dir / EDA_FILE).open("w", encoding="utf-8") as handle:
		json.dump(eda, handle, indent=2, sort_keys=True)
		handle.write("\n")
	return quality, eda


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
	parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
	parser.add_argument("--property-sample-rows", type=int, default=PROPERTY_SAMPLE_ROWS)
	arguments = parser.parse_args()
	try:
		quality, eda = build_eda(arguments.raw_dir, arguments.processed_dir, arguments.property_sample_rows)
	except (OSError, ValueError, csv.Error) as error:
		print("Retailrocket EDA failed: {}".format(error), file=sys.stderr)
		return 1
	print(json.dumps({
		"quality_path": str(arguments.processed_dir / QUALITY_FILE),
		"eda_path": str(arguments.processed_dir / EDA_FILE),
		"records_before_cleaning": eda["records_before_cleaning"],
		"records_after_cleaning": eda["records_after_cleaning"],
		"unique_visitors": eda["unique_visitors"],
		"unique_items": eda["unique_items"],
	}, indent=2))
	return 0


if __name__ == "__main__":
	sys.exit(main())
