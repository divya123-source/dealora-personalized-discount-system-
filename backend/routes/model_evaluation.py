"""Read-only API that exposes the REAL evaluation results of the recommendation
experiment (data/processed/trial/*.json) so the dashboard can display them.

Nothing here is hardcoded: every number comes from the experiment files.
Note: the experiment used Retailrocket IDs, not DealoRa product IDs, so this
endpoint reports evaluation results only. It does not generate live product
recommendations.
"""
import json
import os

from flask import Blueprint, jsonify

model_evaluation_bp = Blueprint("model_evaluation", __name__)

TRIAL_DIR = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "trial")
)
EXPERIMENT_FILE = os.path.join(TRIAL_DIR, "retailrocket_experiment.json")
MODEL_CARD_FILE = os.path.join(TRIAL_DIR, "model_card.json")

RANKING_KEYS = (
	"evaluation_users",
	"hit_rate_at_5", "hit_rate_at_10",
	"ndcg_at_5", "ndcg_at_10",
	"map_at_5", "map_at_10",
	"precision_at_5", "precision_at_10",
	"recall_at_5", "recall_at_10",
)


def _load(path):
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _final_summary(block):
	summary = {key: block.get(key) for key in RANKING_KEYS}
	classification = block.get("sampled_candidate_classification") or {}
	summary["sampled_candidate_roc_auc"] = classification.get("roc_auc")
	return summary


@model_evaluation_bp.get("/api/model/evaluation")
def model_evaluation():
	try:
		experiment = _load(EXPERIMENT_FILE)
		card = _load(MODEL_CARD_FILE)
	except FileNotFoundError as error:
		return jsonify({
			"error": "Evaluation files not found",
			"detail": os.path.basename(error.filename or ""),
		}), 503
	except (ValueError, OSError):
		return jsonify({"error": "Evaluation files could not be read"}), 500

	protocol = experiment.get("protocol", {})
	target = experiment.get("target_85_90", {})
	final_test = experiment.get("final_test", {})

	candidates = []
	for item in experiment.get("candidates", []):
		cv = item.get("cv_summary", {})
		candidates.append({
			"name": item.get("candidate", {}).get("name"),
			"model": item.get("candidate", {}).get("model"),
			"neighbors": item.get("candidate", {}).get("neighbors"),
			"recent_window": item.get("candidate", {}).get("recent_window"),
			"folds": len(item.get("folds", [])),
			"cv": {
				metric: {"mean": stats.get("mean"), "std": stats.get("std")}
				for metric, stats in cv.items()
			},
		})

	return jsonify({
		"model": {
			"name": card.get("model_name"),
			"version": card.get("model_version"),
			"built_at_utc": card.get("built_at_utc"),
			"hyperparameters": card.get("hyperparameters"),
			"features": card.get("features"),
			"id_namespace": card.get("id_namespace"),
			"used_in_live_store": not card.get("not_used_in_live_store", True),
		},
		"dataset": experiment.get("dataset", {}),
		"protocol": {
			"split": protocol.get("split"),
			"cross_validation": protocol.get("cross_validation"),
			"selection_metric": protocol.get("selection_metric"),
			"k_values": protocol.get("k_values"),
		},
		"selection": {
			"selected_model": experiment.get("selected_model"),
			"reason": experiment.get("selection_reason"),
		},
		"candidates": candidates,
		"final_test": {
			"popularity_baseline": _final_summary(final_test.get("popularity_baseline", {})),
			"selected_model": _final_summary(final_test.get("selected_model", {})),
		},
		"target": {
			"requested": target.get("requested"),
			"primary_reported_metric": target.get("primary_reported_metric"),
			"selected_model_ndcg_at_10": target.get("selected_model_ndcg_at_10"),
			"target_met": target.get("target_met"),
			"note": protocol.get("target_note"),
			"limitations": target.get("limitations", []),
			"legitimate_improvements": target.get("legitimate_improvements", []),
		},
		"generated_at_utc": experiment.get("generated_at_utc"),
	})