"""Locate offline Retailrocket artifacts without touching raw files or leftover temp dirs."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_FILES = {
	"evaluation": "retailrocket_evaluation.json",
	"experiment": "retailrocket_experiment.json",
	"quality": "data_quality_report.json",
	"eda": "eda_summary.json",
	"model_card": "model_card.json",
}


def processed_dir():
	return PROCESSED_DIR


def leftover_temp_dirs():
	if not PROCESSED_DIR.is_dir():
		return []
	return sorted(
		path.name
		for path in PROCESSED_DIR.iterdir()
		if path.is_dir() and path.name.startswith(".retailrocket-")
	)


def load_json_if_present(name):
	path = PROCESSED_DIR / name
	if not path.is_file():
		return None
	with path.open(encoding="utf-8") as handle:
		return json.load(handle)


def artifact_status():
	status = {
		filename: (PROCESSED_DIR / filename).is_file()
		for filename in (
			"interactions.csv",
			"category_tree.csv",
			"item_properties.csv",
			"validation_report.json",
			"retailrocket_recommendation_model.sqlite3",
			*REPORT_FILES.values(),
		)
	}
	return {
		"processed_dir": str(PROCESSED_DIR),
		"files": status,
		"leftover_incomplete_temp_dirs": leftover_temp_dirs(),
		"leftover_temp_dirs_preserved": True,
		"id_namespace": "Retailrocket visitorid/itemid; not DealoRa database IDs",
	}


def load_offline_bundle():
	return {
		"status": artifact_status(),
		"quality": load_json_if_present(REPORT_FILES["quality"]),
		"eda": load_json_if_present(REPORT_FILES["eda"]),
		"experiment": load_json_if_present(REPORT_FILES["experiment"]),
		"evaluation": load_json_if_present(REPORT_FILES["evaluation"]),
		"model_card": load_json_if_present(REPORT_FILES["model_card"]),
	}
