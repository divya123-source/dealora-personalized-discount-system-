"""Walk-forward CV, baseline comparison, and held-out test for Retailrocket recommenders."""

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
	classification_metrics,
	mean_metrics,
	metric_mean_std,
	ranking_metrics,
	roc_auc_score,
	threshold_predictions,
)
from train_retailrocket import (  # noqa: E402
	DEFAULT_DATA_DIR,
	EVENT_WEIGHTS,
	INTERACTIONS_FILE,
	MODEL_FILE,
	RESULTS_FILE,
	_load_interactions,
	_publish_outputs,
	train_evaluate,
)


EXPERIMENT_FILE = "retailrocket_experiment.json"
MODEL_CARD_FILE = "model_card.json"
CANDIDATES = (
	{"name": "popularity_baseline", "model": "popularity", "recent_window": 0, "neighbors": 0},
	{"name": "covisitation_w3_n50", "model": "covisitation", "recent_window": 3, "neighbors": 50},
	{"name": "covisitation_w5_n100", "model": "covisitation", "recent_window": 5, "neighbors": 100},
	{"name": "covisitation_w10_n100", "model": "covisitation", "recent_window": 10, "neighbors": 100},
)


def _cutoff_at_fraction(database, fraction):
	total = database.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
	if total < 4:
		raise ValueError("Need at least 4 interactions for a time-ordered train/validation/test split.")
	offset = min(max(int(total * fraction), 1), total - 1)
	return database.execute(
		"SELECT timestamp_ms FROM interactions ORDER BY timestamp_ms, rowid LIMIT 1 OFFSET ?",
		(offset,),
	).fetchone()[0]


def _build_tables(database, train_end, recent_window):
	database.execute("DROP TABLE IF EXISTS popularity")
	database.execute("DROP TABLE IF EXISTS co_visitation")
	database.execute("DROP TABLE IF EXISTS user_recent")
	database.execute("DROP TABLE IF EXISTS train_purchases")
	database.execute(
		"CREATE TABLE popularity (item_id INTEGER PRIMARY KEY, score REAL NOT NULL) WITHOUT ROWID"
	)
	database.execute(
		"INSERT INTO popularity(item_id, score) "
		"SELECT item_id, SUM(CASE event_type WHEN 'purchase' THEN 3.0 "
		"WHEN 'cart' THEN 2.0 ELSE 1.0 END) "
		"FROM interactions WHERE timestamp_ms < ? GROUP BY item_id",
		(train_end,),
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
		(train_end,),
	)
	pair_upsert = (
		"INSERT INTO co_visitation(source_item_id, target_item_id, score) VALUES (?, ?, ?) "
		"ON CONFLICT(source_item_id, target_item_id) DO UPDATE SET score = score + excluded.score"
	)
	pair_batch = []
	recent_batch = []
	current_visitor = None
	history = deque(maxlen=max(int(recent_window), 1))
	cursor = database.execute(
		"SELECT visitor_id, item_id, event_type FROM interactions "
		"WHERE timestamp_ms < ? ORDER BY visitor_id, timestamp_ms, rowid",
		(train_end,),
	)
	for visitor_id, item_id, event_type in cursor:
		if current_visitor is not None and visitor_id != current_visitor:
			recent_batch.append((current_visitor, json.dumps(list(history))))
			history.clear()
		if visitor_id != current_visitor:
			current_visitor = visitor_id
		if recent_window:
			for distance, source_item_id in enumerate(reversed(history), start=1):
				if source_item_id != item_id:
					pair_batch.append((source_item_id, item_id, EVENT_WEIGHTS[event_type] / distance))
		history.append(item_id)
		if len(pair_batch) >= 50000:
			database.executemany(pair_upsert, pair_batch)
			pair_batch.clear()
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


def _relevant_users(database, start_ms, end_ms, exclusive_end):
	end_clause = "timestamp_ms < ?" if exclusive_end else "timestamp_ms >= ?"
	query = (
		"SELECT DISTINCT visitor_id FROM interactions "
		"WHERE event_type = 'purchase' AND timestamp_ms >= ? AND {} "
		"ORDER BY visitor_id"
	).format(end_clause)
	return [row[0] for row in database.execute(query, (start_ms, end_ms))]


def _relevant_items(database, visitor_id, start_ms, end_ms, exclusive_end):
	end_clause = "timestamp_ms < ?" if exclusive_end else "timestamp_ms >= ?"
	query = (
		"SELECT item_id FROM interactions WHERE visitor_id = ? AND event_type = 'purchase' "
		"AND timestamp_ms >= ? AND {} AND NOT EXISTS ("
		"SELECT 1 FROM train_purchases WHERE train_purchases.visitor_id = ? "
		"AND train_purchases.item_id = interactions.item_id)"
	).format(end_clause)
	return {row[0] for row in database.execute(query, (visitor_id, start_ms, end_ms, visitor_id))}


def _recommend(database, visitor_id, k, model, neighbors, popularity_fallback):
	purchased = {
		item_id for (item_id,) in database.execute(
			"SELECT item_id FROM train_purchases WHERE visitor_id = ?", (visitor_id,)
		)
	}
	scores = {}
	if model == "covisitation":
		row = database.execute(
			"SELECT items_json FROM user_recent WHERE visitor_id = ?", (visitor_id,)
		).fetchone()
		recent_items = json.loads(row[0]) if row else []
		for recency, source_item_id in enumerate(reversed(recent_items), start=1):
			neighbors_rows = database.execute(
				"SELECT target_item_id, score FROM co_visitation "
				"WHERE source_item_id = ? ORDER BY score DESC, target_item_id ASC LIMIT ?",
				(source_item_id, max(int(neighbors), 1)),
			)
			for target_item_id, score in neighbors_rows:
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
	return selected, scores


def _evaluate_split(database, k, ks, model, neighbors, start_ms, end_ms, exclusive_end, max_eval_users, seed):
	user_ids = _relevant_users(database, start_ms, end_ms, exclusive_end)
	if max_eval_users and len(user_ids) > max_eval_users:
		step = max(len(user_ids) // max_eval_users, 1)
		user_ids = user_ids[::step][:max_eval_users]
	popularity_fallback = database.execute(
		"SELECT item_id, score FROM popularity ORDER BY score DESC, item_id ASC LIMIT ?",
		(max(500, k * 50),),
	).fetchall()
	rows_by_k = {value: [] for value in ks}
	class_labels = []
	class_scores = []
	evaluated = 0
	for visitor_id in user_ids:
		relevant = _relevant_items(database, visitor_id, start_ms, end_ms, exclusive_end)
		if not relevant:
			continue
		recommended, scores = _recommend(database, visitor_id, max(ks), model, neighbors, popularity_fallback)
		for value in ks:
			rows_by_k[value].append(ranking_metrics(recommended, relevant, value))
		negatives = [item_id for item_id, _score in popularity_fallback if item_id not in relevant][:20]
		for item_id in list(relevant)[:20]:
			class_labels.append(True)
			class_scores.append(float(scores.get(item_id, 0.0)))
		for item_id in negatives:
			class_labels.append(False)
			class_scores.append(float(scores.get(item_id, 0.0)))
		evaluated += 1
	aggregated = {}
	for value in ks:
		means = mean_metrics(rows_by_k[value])
		aggregated["precision_at_{}".format(value)] = means["precision"]
		aggregated["recall_at_{}".format(value)] = means["recall"]
		aggregated["ndcg_at_{}".format(value)] = means["ndcg"]
		aggregated["map_at_{}".format(value)] = means["map"]
		aggregated["hit_rate_at_{}".format(value)] = means["hit_rate"]
	aggregated["evaluation_users"] = evaluated
	threshold = sorted(class_scores)[len(class_scores) // 2] if class_scores else 0.0
	predictions = threshold_predictions(class_scores, threshold)
	aggregated["sampled_candidate_classification"] = classification_metrics(class_labels, predictions)
	aggregated["sampled_candidate_classification"]["roc_auc"] = roc_auc_score(class_labels, class_scores)
	aggregated["sampled_candidate_classification"]["decision_threshold"] = threshold
	aggregated["sampled_candidate_classification"]["seed"] = seed
	return aggregated


def _summarize_folds(fold_rows, keys):
	summary = {}
	for key in keys:
		summary[key] = metric_mean_std([row.get(key) for row in fold_rows])
	return summary


def run_experiment(
	interactions_path=DEFAULT_DATA_DIR / INTERACTIONS_FILE,
	output_dir=DEFAULT_DATA_DIR,
	k=10,
	max_eval_users=1500,
	write_serving_model=True,
):
	interactions_path = Path(interactions_path)
	output_dir = Path(output_dir)
	ks = (5, 10)
	output_dir.mkdir(parents=True, exist_ok=True)
	existing = [name for name in (EXPERIMENT_FILE, MODEL_CARD_FILE) if (output_dir / name).exists()]
	if existing:
		raise FileExistsError("Refusing to overwrite output(s): {}".format(", ".join(existing)))
	if not interactions_path.is_file():
		raise FileNotFoundError("Processed interactions not found: {}".format(interactions_path))

	with tempfile.TemporaryDirectory(prefix=".retailrocket-exp-", dir=str(output_dir)) as temp_dir:
		database = sqlite3.connect(str(Path(temp_dir) / "experiment.sqlite3"))
		database.execute(
			"CREATE TABLE interactions (visitor_id INTEGER NOT NULL, item_id INTEGER NOT NULL, "
			"event_type TEXT NOT NULL, timestamp_ms INTEGER NOT NULL)"
		)
		database.execute("CREATE INDEX interactions_by_time ON interactions(timestamp_ms)")
		load_report = _load_interactions(interactions_path, database)
		test_start = _cutoff_at_fraction(database, 0.8)
		fold_bounds = [
			(_cutoff_at_fraction(database, 0.50), _cutoff_at_fraction(database, 0.65)),
			(_cutoff_at_fraction(database, 0.65), _cutoff_at_fraction(database, 0.80)),
		]
		cv_results = []
		for candidate in CANDIDATES:
			fold_metrics = []
			for index, (train_end, val_end) in enumerate(fold_bounds, start=1):
				_build_tables(database, train_end, candidate["recent_window"])
				metrics = _evaluate_split(
					database, k, ks, candidate["model"], candidate["neighbors"],
					train_end, val_end, True, max_eval_users, seed=index,
				)
				fold_metrics.append({"fold": index, "train_end_timestamp_ms": train_end, "val_end_timestamp_ms": val_end, **metrics})
			cv_results.append({
				"candidate": candidate,
				"folds": fold_metrics,
				"cv_summary": _summarize_folds(
					fold_metrics,
					("ndcg_at_10", "precision_at_5", "precision_at_10", "recall_at_10", "map_at_10"),
				),
			})

		def score(row):
			mean = row["cv_summary"]["ndcg_at_10"]["mean"]
			return (mean is not None, mean or -1.0)

		winner = max(cv_results, key=score)
		_build_tables(database, test_start, winner["candidate"]["recent_window"])
		test_metrics = _evaluate_split(
			database, k, ks, winner["candidate"]["model"], winner["candidate"]["neighbors"],
			test_start, test_start, False, max_eval_users, seed=99,
		)
		_build_tables(database, test_start, 0)
		baseline_test = _evaluate_split(
			database, k, ks, "popularity", 0, test_start, test_start, False, max_eval_users, seed=99,
		)
		database.close()

		target = 0.85
		achieved = test_metrics.get("ndcg_at_10")
		experiment = {
			"generated_at_utc": datetime.now(timezone.utc).isoformat(),
			"id_namespace": "Retailrocket visitorid/itemid; not DealoRa database IDs",
			"dataset": {
				"interactions_csv": str(interactions_path),
				"valid_rows": load_report["input_rows"] - load_report["rejected_rows"],
				"rejected_rows": load_report["rejected_rows"],
			},
			"protocol": {
				"split": "global chronological 80/20 holdout; test timestamps never used in training, CV, or tuning",
				"cross_validation": "two walk-forward folds on the training period only (50-65% and 65-80%)",
				"selection_metric": "mean NDCG@10 across CV folds",
				"max_eval_users_per_split": max_eval_users,
				"k_values": list(ks),
				"target_note": (
					"An 85-90% Precision@K on a large implicit-feedback catalog is not a realistic "
					"Retailrocket ranking target. Reported ranking metrics are measured. Classification "
					"metrics use sampled candidates and are not catalog-wide accuracy."
				),
			},
			"candidates": cv_results,
			"selected_model": winner["candidate"],
			"selection_reason": (
				"Selected {name} because it had the highest mean NDCG@10 across time-ordered "
				"validation folds ({mean})."
			).format(
				name=winner["candidate"]["name"],
				mean=winner["cv_summary"]["ndcg_at_10"]["mean"],
			),
			"final_test": {
				"selected_model": test_metrics,
				"popularity_baseline": baseline_test,
			},
			"target_85_90": {
				"requested": "85-90% recommendation performance",
				"primary_reported_metric": "ndcg_at_10",
				"selected_model_ndcg_at_10": achieved,
				"target_met": bool(achieved is not None and achieved >= target),
				"limitations": [
					"Retailrocket purchases are sparse relative to the item catalog, so Precision@K is typically low.",
					"HitRate@K and NDCG@K are the more informative ranking measures for this task.",
					"Classification accuracy can look high from class imbalance; ROC-AUC on sampled candidates is preferred.",
				],
				"legitimate_improvements": [
					"Session-based sequential models (GRU4Rec, SASRec) on GPU.",
					"Content features from decoded item properties, if a verified mapping exists.",
					"Larger evaluation user samples once more RAM/time is available.",
				],
			},
		}
		model_card = {
			"model_name": winner["candidate"]["name"],
			"model_version": "2.0",
			"built_at_utc": experiment["generated_at_utc"],
			"features": [
				"visitor_id",
				"item_id",
				"event_type (view/cart/purchase)",
				"timestamp_ms",
				"directed co-visitation within a recent-item window",
				"train-period weighted item popularity",
			],
			"hyperparameters": winner["candidate"],
			"id_namespace": experiment["id_namespace"],
			"not_used_in_live_store": True,
		}
		with (Path(temp_dir) / EXPERIMENT_FILE).open("w", encoding="utf-8") as handle:
			json.dump(experiment, handle, indent=2, sort_keys=True)
			handle.write("\n")
		with (Path(temp_dir) / MODEL_CARD_FILE).open("w", encoding="utf-8") as handle:
			json.dump(model_card, handle, indent=2, sort_keys=True)
			handle.write("\n")
		_publish_outputs(temp_dir, output_dir, (EXPERIMENT_FILE, MODEL_CARD_FILE))

	if write_serving_model and not (output_dir / MODEL_FILE).exists() and not (output_dir / RESULTS_FILE).exists():
		if winner["candidate"]["model"] == "covisitation":
			train_evaluate(interactions_path, output_dir, k=k)
	return experiment


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--interactions", type=Path, default=DEFAULT_DATA_DIR / INTERACTIONS_FILE)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--k", type=int, default=10)
	parser.add_argument("--max-eval-users", type=int, default=1500)
	parser.add_argument("--skip-serving-model", action="store_true")
	arguments = parser.parse_args()
	try:
		results = run_experiment(
			arguments.interactions,
			arguments.output_dir,
			arguments.k,
			arguments.max_eval_users,
			write_serving_model=not arguments.skip_serving_model,
		)
	except (OSError, ValueError, sqlite3.Error, csv.Error) as error:
		print("Retailrocket experiment failed: {}".format(error), file=sys.stderr)
		return 1
	print(json.dumps({
		"selected_model": results["selected_model"],
		"selection_reason": results["selection_reason"],
		"final_test": results["final_test"]["selected_model"],
		"target_met": results["target_85_90"]["target_met"],
		"experiment_path": str(arguments.output_dir / EXPERIMENT_FILE),
	}, indent=2))
	return 0


if __name__ == "__main__":
	sys.exit(main())
