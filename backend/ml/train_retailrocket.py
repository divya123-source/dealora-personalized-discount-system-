"""Train and evaluate an offline Retailrocket co-visitation recommender."""

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INTERACTIONS_FILE = "interactions.csv"
MODEL_FILE = "retailrocket_recommendation_model.sqlite3"
RESULTS_FILE = "retailrocket_evaluation.json"
MODEL_NAME = "Retailrocket sequential co-visitation and popularity"
MODEL_VERSION = "1.0"
EVENT_WEIGHTS = {"view": 1.0, "cart": 2.0, "purchase": 3.0}
RECENT_WINDOW = 5
NEIGHBORS_PER_ITEM = 100


def ranking_metrics(recommended, relevant, k):
	"""Return fixed-slot Precision@K, Recall@K, and binary NDCG@K."""
	if k < 1:
		raise ValueError("k must be positive")
	relevant = set(relevant)
	selected = list(recommended[:k])
	hits = sum(item_id in relevant for item_id in selected)
	precision = hits / float(k)
	recall = hits / float(len(relevant)) if relevant else 0.0
	dcg = sum(
		1.0 / math.log2(rank + 2)
		for rank, item_id in enumerate(selected)
		if item_id in relevant
	)
	ideal_hits = min(k, len(relevant))
	idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
	ndcg = dcg / idcg if idcg else 0.0
	return {"precision": precision, "recall": recall, "ndcg": ndcg}


def _publish_outputs(temporary_dir, output_dir):
	created = []
	try:
		for filename in (MODEL_FILE, RESULTS_FILE):
			source = Path(temporary_dir) / filename
			destination = output_dir / filename
			os.link(str(source), str(destination))
			created.append(destination)
	except OSError:
		for destination in created:
			destination.unlink()
		raise


def _load_interactions(path, database):
	stats = {"input_rows": 0, "rejected_rows": 0, "errors": {}}
	with path.open("r", encoding="utf-8-sig", newline="") as handle:
		reader = csv.DictReader(handle)
		required = {"visitor_id", "item_id", "event_type", "timestamp_ms"}
		if reader.fieldnames is None or not required.issubset(reader.fieldnames):
			missing = sorted(required - set(reader.fieldnames or ()))
			raise ValueError("Interactions CSV is missing required columns: {}".format(", ".join(missing)))
		batch = []
		error_counts = {}
		for row in reader:
			stats["input_rows"] += 1
			try:
				visitor_id = int(row["visitor_id"])
				item_id = int(row["item_id"])
				timestamp_ms = int(row["timestamp_ms"])
				event_type = (row["event_type"] or "").strip().lower()
				if visitor_id < 0 or item_id < 0 or timestamp_ms < 0:
					raise ValueError("negative identifier or timestamp")
				if event_type not in EVENT_WEIGHTS:
					raise ValueError("unsupported event type")
			except (TypeError, ValueError, KeyError):
				stats["rejected_rows"] += 1
				reason = "invalid_interaction_row"
				error_counts[reason] = error_counts.get(reason, 0) + 1
				continue
			batch.append((visitor_id, item_id, event_type, timestamp_ms))
			if len(batch) >= 50000:
				database.executemany(
					"INSERT INTO interactions(visitor_id, item_id, event_type, timestamp_ms) VALUES (?, ?, ?, ?)",
					batch,
				)
				batch.clear()
		if batch:
			database.executemany(
				"INSERT INTO interactions(visitor_id, item_id, event_type, timestamp_ms) VALUES (?, ?, ?, ?)",
				batch,
			)
	database.commit()
	stats["errors"] = error_counts
	return stats


def _prepare_training_data(database, k):
	total_rows = database.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
	if total_rows < 2:
		raise ValueError("At least two valid interactions are required for a time split.")
	cutoff_offset = int(total_rows * 0.8)
	cutoff = database.execute(
		"SELECT timestamp_ms FROM interactions ORDER BY timestamp_ms, rowid LIMIT 1 OFFSET ?",
		(cutoff_offset,),
	).fetchone()[0]
	train_rows = database.execute(
		"SELECT COUNT(*) FROM interactions WHERE timestamp_ms < ?", (cutoff,)
	).fetchone()[0]
	test_rows = database.execute(
		"SELECT COUNT(*) FROM interactions WHERE timestamp_ms >= ?", (cutoff,)
	).fetchone()[0]
	if not train_rows or not test_rows:
		raise ValueError(
			"The chronological boundary leaves an empty train or test partition; "
			"more distinct timestamps are needed."
		)
	database.execute(
		"CREATE TABLE popularity (item_id INTEGER PRIMARY KEY, score REAL NOT NULL) WITHOUT ROWID"
	)
	database.execute(
		"INSERT INTO popularity(item_id, score) "
		"SELECT item_id, SUM(CASE event_type WHEN 'purchase' THEN 3.0 "
		"WHEN 'cart' THEN 2.0 ELSE 1.0 END) "
		"FROM interactions WHERE timestamp_ms < ? GROUP BY item_id",
		(cutoff,),
	)
	database.execute(
		"CREATE TABLE co_visitation (source_item_id INTEGER NOT NULL, target_item_id INTEGER NOT NULL, "
		"score REAL NOT NULL, PRIMARY KEY(source_item_id, target_item_id)) WITHOUT ROWID"
	)
	database.execute(
		"CREATE TABLE user_recent (visitor_id INTEGER PRIMARY KEY, items_json TEXT NOT NULL) WITHOUT ROWID"
	)
	database.execute(
		"CREATE TABLE train_purchases (visitor_id INTEGER NOT NULL, item_id INTEGER NOT NULL, "
		"PRIMARY KEY(visitor_id, item_id)) WITHOUT ROWID"
	)
	database.execute(
		"INSERT OR IGNORE INTO train_purchases(visitor_id, item_id) "
		"SELECT visitor_id, item_id FROM interactions "
		"WHERE timestamp_ms < ? AND event_type = 'purchase'",
		(cutoff,),
	)
	database.execute(
		"CREATE TABLE relevance (visitor_id INTEGER NOT NULL, item_id INTEGER NOT NULL, "
		"PRIMARY KEY(visitor_id, item_id)) WITHOUT ROWID"
	)
	database.execute(
		"INSERT OR IGNORE INTO relevance(visitor_id, item_id) "
		"SELECT test.visitor_id, test.item_id FROM interactions AS test "
		"WHERE test.timestamp_ms >= ? AND test.event_type = 'purchase' "
		"AND NOT EXISTS (SELECT 1 FROM train_purchases AS past "
		"WHERE past.visitor_id = test.visitor_id AND past.item_id = test.item_id)",
		(cutoff,),
	)
	database.execute("CREATE INDEX interactions_by_user_time ON interactions(visitor_id, timestamp_ms)")
	database.execute("CREATE INDEX neighbors_by_score ON co_visitation(source_item_id, score DESC, target_item_id)")

	pair_upsert = (
		"INSERT INTO co_visitation(source_item_id, target_item_id, score) VALUES (?, ?, ?) "
		"ON CONFLICT(source_item_id, target_item_id) DO UPDATE SET score = score + excluded.score"
	)
	pair_batch = []
	recent_batch = []
	current_visitor = None
	history = deque(maxlen=RECENT_WINDOW)
	cursor = database.execute(
		"SELECT visitor_id, item_id, event_type FROM interactions "
		"WHERE timestamp_ms < ? ORDER BY visitor_id, timestamp_ms, rowid",
		(cutoff,),
	)
	for visitor_id, item_id, event_type in cursor:
		if current_visitor is not None and visitor_id != current_visitor:
			recent_batch.append((current_visitor, json.dumps(list(history))))
			history.clear()
		if visitor_id != current_visitor:
			current_visitor = visitor_id
		for distance, source_item_id in enumerate(reversed(history), start=1):
			if source_item_id != item_id:
				pair_batch.append((source_item_id, item_id, EVENT_WEIGHTS[event_type] / distance))
		history.append(item_id)
		if len(pair_batch) >= 50000:
			database.executemany(pair_upsert, pair_batch)
			pair_batch.clear()
			database.commit()
		if len(recent_batch) >= 10000:
			database.executemany(
				"INSERT INTO user_recent(visitor_id, items_json) VALUES (?, ?)", recent_batch
			)
			recent_batch.clear()
	if current_visitor is not None:
		recent_batch.append((current_visitor, json.dumps(list(history))))
	if pair_batch:
		database.executemany(pair_upsert, pair_batch)
	if recent_batch:
		database.executemany(
			"INSERT INTO user_recent(visitor_id, items_json) VALUES (?, ?)", recent_batch
		)
	database.commit()

	relevance_users = database.execute("SELECT COUNT(DISTINCT visitor_id) FROM relevance").fetchone()[0]
	cold_start_users = database.execute(
		"SELECT COUNT(DISTINCT relevance.visitor_id) FROM relevance "
		"LEFT JOIN user_recent USING(visitor_id) WHERE user_recent.visitor_id IS NULL"
	).fetchone()[0]
	return {
		"input_rows": total_rows,
		"train_interactions": train_rows,
		"test_interactions": test_rows,
		"cutoff_timestamp_ms": cutoff,
		"cutoff_timestamp_utc": datetime.fromtimestamp(cutoff / 1000, timezone.utc).isoformat(),
		"test_purchase_relevant_items": database.execute("SELECT COUNT(*) FROM relevance").fetchone()[0],
		"evaluation_users": relevance_users,
		"cold_start_evaluation_users": cold_start_users,
		"k": k,
	}


def _recommend_for_user(database, visitor_id, k, popularity_fallback):
	row = database.execute(
		"SELECT items_json FROM user_recent WHERE visitor_id = ?", (visitor_id,)
	).fetchone()
	recent_items = json.loads(row[0]) if row else []
	purchased = {
		item_id for (item_id,) in database.execute(
			"SELECT item_id FROM train_purchases WHERE visitor_id = ?", (visitor_id,)
		)
	}
	scores = {}
	for recency, source_item_id in enumerate(reversed(recent_items), start=1):
		neighbors = database.execute(
			"SELECT target_item_id, score FROM co_visitation "
			"WHERE source_item_id = ? ORDER BY score DESC, target_item_id ASC LIMIT ?",
			(source_item_id, NEIGHBORS_PER_ITEM),
		)
		for target_item_id, score in neighbors:
			if target_item_id not in purchased:
				scores[target_item_id] = scores.get(target_item_id, 0.0) + score / recency

	popularity = dict(popularity_fallback)
	ranked = sorted(
		scores,
		key=lambda item_id: (-scores[item_id], -popularity.get(item_id, 0.0), item_id),
	)
	selected = [item_id for item_id in ranked if item_id not in purchased][:k]
	if len(selected) < k:
		selected_set = set(selected)
		for item_id, _score in popularity_fallback:
			if item_id in purchased or item_id in selected_set:
				continue
			selected.append(item_id)
			selected_set.add(item_id)
			if len(selected) == k:
				break
	return selected


def _evaluate(database, k):
	user_ids = [
		row[0] for row in database.execute(
			"SELECT DISTINCT visitor_id FROM relevance ORDER BY visitor_id"
		)
	]
	popularity_fallback = database.execute(
		"SELECT item_id, score FROM popularity ORDER BY score DESC, item_id ASC LIMIT ?",
		(max(500, k * 50),),
	).fetchall()
	metric_sums = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
	evaluated = 0
	for visitor_id in user_ids:
		relevant = {
			item_id for (item_id,) in database.execute(
				"SELECT item_id FROM relevance WHERE visitor_id = ?", (visitor_id,)
			)
		}
		if not relevant:
			continue
		recommended = _recommend_for_user(database, visitor_id, k, popularity_fallback)
		metrics = ranking_metrics(recommended, relevant, k)
		for name, value in metrics.items():
			metric_sums[name] += value
		evaluated += 1
	if not evaluated:
		return {"evaluation_users": 0, "precision_at_k": None, "recall_at_k": None, "ndcg_at_k": None}
	return {
		"evaluation_users": evaluated,
		"precision_at_k": metric_sums["precision"] / evaluated,
		"recall_at_k": metric_sums["recall"] / evaluated,
		"ndcg_at_k": metric_sums["ndcg"] / evaluated,
	}


def train_evaluate(interactions_path=DEFAULT_DATA_DIR / INTERACTIONS_FILE, output_dir=DEFAULT_DATA_DIR, k=10):
	"""Create offline-only model and actual temporal-holdout metrics; never overwrite outputs."""
	interactions_path = Path(interactions_path)
	output_dir = Path(output_dir)
	if k < 1:
		raise ValueError("k must be positive")
	if not interactions_path.is_file():
		raise FileNotFoundError("Processed interactions not found: {}".format(interactions_path))
	output_dir.mkdir(parents=True, exist_ok=True)
	existing = [name for name in (MODEL_FILE, RESULTS_FILE) if (output_dir / name).exists()]
	if existing:
		raise FileExistsError("Refusing to overwrite output(s): {}".format(", ".join(existing)))

	with tempfile.TemporaryDirectory(prefix=".retailrocket-train-", dir=str(output_dir)) as temp_dir:
		temp_dir = Path(temp_dir)
		staging_path = temp_dir / "staging.sqlite3"
		database = sqlite3.connect(str(staging_path))
		database.execute(
			"CREATE TABLE interactions (visitor_id INTEGER NOT NULL, item_id INTEGER NOT NULL, "
			"event_type TEXT NOT NULL, timestamp_ms INTEGER NOT NULL)"
		)
		database.execute("CREATE INDEX interactions_by_time ON interactions(timestamp_ms)")
		load_report = _load_interactions(interactions_path, database)
		if not load_report["input_rows"]:
			database.close()
			raise ValueError("The processed interactions file contains no valid rows.")
		split_report = _prepare_training_data(database, k)
		metrics = _evaluate(database, k)
		model_path = temp_dir / MODEL_FILE
		model_db = sqlite3.connect(str(model_path))
		model_db.executescript(
			"CREATE TABLE model_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
			"CREATE TABLE item_popularity (item_id INTEGER PRIMARY KEY, score REAL NOT NULL);"
			"CREATE TABLE item_neighbors (source_item_id INTEGER NOT NULL, target_item_id INTEGER NOT NULL, "
			"score REAL NOT NULL, PRIMARY KEY(source_item_id, target_item_id));"
			"CREATE INDEX item_neighbors_by_score ON item_neighbors(source_item_id, score DESC, target_item_id);"
			"CREATE TABLE user_recent (visitor_id INTEGER PRIMARY KEY, items_json TEXT NOT NULL);"
			"CREATE TABLE train_purchases (visitor_id INTEGER NOT NULL, item_id INTEGER NOT NULL, "
			"PRIMARY KEY(visitor_id, item_id));"
		)
		model_db.execute("ATTACH DATABASE ? AS training", (str(staging_path),))
		model_db.execute("INSERT INTO item_popularity SELECT item_id, score FROM training.popularity")
		model_db.execute(
			"INSERT INTO item_neighbors SELECT source_item_id, target_item_id, score FROM training.co_visitation"
		)
		model_db.execute("INSERT INTO user_recent SELECT visitor_id, items_json FROM training.user_recent")
		model_db.execute("INSERT INTO train_purchases SELECT visitor_id, item_id FROM training.train_purchases")
		metadata = {
			"model_name": MODEL_NAME,
			"model_version": MODEL_VERSION,
			"built_at_utc": datetime.now(timezone.utc).isoformat(),
			"id_namespace": "Retailrocket visitorid/itemid; not DealoRa database IDs",
			"recommendation_method": "last-five-item directed co-visitation with train-only weighted popularity fallback",
			"training_cutoff_timestamp_ms": split_report["cutoff_timestamp_ms"],
			"k": k,
		}
		model_db.executemany(
			"INSERT INTO model_metadata(key, value) VALUES (?, ?)",
			[(key, json.dumps(value)) for key, value in metadata.items()],
		)
		model_db.commit()
		model_db.execute("DETACH DATABASE training")
		model_db.close()
		database.close()

		results = {
			"model_name": MODEL_NAME,
			"model_version": MODEL_VERSION,
			"evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
			"evaluation_method": {
				"split": "global chronological 80/20 holdout; all rows at the cutoff timestamp are assigned to test",
				"relevance": "distinct test-period purchase items not purchased by that visitor before the cutoff",
				"candidate_ranking": "directed co-visitation scores from each visitor's last five training interactions; training-period popularity fills remaining slots",
				"exclusions": "items purchased by that visitor in training are excluded; test events never train the model",
				"aggregation": "macro average over visitors with at least one eligible test purchase",
				"precision_definition": "hits in top K divided by K, including unfilled slots",
				"ndcg_definition": "binary relevance with logarithmic rank discount and ideal DCG over min(K, relevant items)",
				"cold_start": "visitors without training interactions receive the same training-only popularity fallback and remain in aggregate metrics",
				"cross_validation": "not used because random folds would leak future interactions across time",
			},
			"input": {
				"processed_interactions_csv": str(interactions_path),
				"valid_rows": load_report["input_rows"] - load_report["rejected_rows"],
				"rejected_rows": load_report["rejected_rows"],
				"processing_errors": load_report["errors"],
			},
			"split": split_report,
			"metrics": metrics,
			"artifacts": {"model": MODEL_FILE},
			"id_namespace": "Retailrocket IDs remain offline and are never mapped to DealoRa IDs",
		}
		with (temp_dir / RESULTS_FILE).open("w", encoding="utf-8") as handle:
			json.dump(results, handle, indent=2, sort_keys=True)
			handle.write("\n")
		_publish_outputs(temp_dir, output_dir)
	return results


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--interactions", type=Path, default=DEFAULT_DATA_DIR / INTERACTIONS_FILE)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--k", type=int, default=10)
	arguments = parser.parse_args()
	try:
		results = train_evaluate(arguments.interactions, arguments.output_dir, arguments.k)
	except (OSError, ValueError, sqlite3.Error, csv.Error) as error:
		print("Retailrocket training/evaluation failed: {}".format(error), file=sys.stderr)
		return 1
	print(json.dumps({
		"model_name": results["model_name"],
		"split": results["split"],
		"metrics": results["metrics"],
		"evaluation_path": str(arguments.output_dir / RESULTS_FILE),
		"model_path": str(arguments.output_dir / MODEL_FILE),
	}, indent=2))
	return 0


if __name__ == "__main__":
	sys.exit(main())