import csv
import json
import tempfile
import unittest
from pathlib import Path

from backend.ml.process_retailrocket import OUTPUT_FILES, _integer, _timestamp, preprocess_retailrocket
from backend.ml.train_retailrocket import (
	MODEL_FILE,
	RESULTS_FILE,
	ranking_metrics,
	train_evaluate,
)


class RetailrocketProcessorTests(unittest.TestCase):
	def setUp(self):
		self.temporary_dir = tempfile.TemporaryDirectory()
		self.root = Path(self.temporary_dir.name)
		self.raw_dir = self.root / "raw"
		self.processed_dir = self.root / "processed"
		self.raw_dir.mkdir()
		self.processed_dir.mkdir()
		self._write_inputs()

	def tearDown(self):
		self.temporary_dir.cleanup()

	def _write(self, name, content):
		(self.raw_dir / name).write_text(content, encoding="utf-8")

	def _write_inputs(self):
		self._write(
			"events.csv",
			"timestamp,visitorid,event,itemid,transactionid\n"
			"1433221332117,10,view,100,\n"
			"1433221332117,10,view,100,\n"
			"1433221333117,10,addtocart,101,\n"
			"1433221334117,11,transaction,102,9001\n"
			"1433221335117,12,transaction,103,\n"
			"1433221336117,12,favorite,104,\n"
			"bad-row\n",
		)
		self._write(
			"category_tree.csv",
			"categoryid,parentid\n"
			"1,\n"
			"1,\n"
			"2,1\n"
			"3,3\n",
		)
		properties = (
			"timestamp,itemid,property,value\n"
			"1435460400000,100,categoryid,1338\n"
			"1435460400000,100,categoryid,1338\n"
			"1435460400001,101,brand,\n"
			"1435460400002,102,,invalid-property\n"
		)
		self._write("item_properties_part1.csv", properties)
		self._write("item_properties_part2.csv", properties)

	def _read_csv(self, filename):
		with (self.processed_dir / filename).open(encoding="utf-8", newline="") as handle:
			return list(csv.DictReader(handle))

	def test_normalizes_deduplicates_and_reports_invalid_rows(self):
		report = preprocess_retailrocket(self.raw_dir, self.processed_dir)
		interactions = self._read_csv("interactions.csv")
		self.assertEqual([row["event_type"] for row in interactions], ["view", "cart", "purchase"])
		self.assertTrue(interactions[0]["timestamp_utc"].startswith("2015-06-02T"))
		self.assertEqual(interactions[2]["transaction_id"], "9001")
		self.assertEqual(report["files"]["events.csv"]["source_rows"], 7)
		self.assertEqual(report["files"]["events.csv"]["output_rows"], 3)
		self.assertEqual(report["files"]["events.csv"]["duplicate_rows"], 1)
		self.assertEqual(report["files"]["events.csv"]["rejected_rows"], 3)
		self.assertEqual(report["files"]["events.csv"]["missing_values"]["visitorid"], 1)
		self.assertEqual(report["unique_visitors"], 2)
		self.assertEqual(report["event_types"]["favorite"], 1)
		self.assertEqual(report["mapped_event_types"], {"cart": 1, "purchase": 1, "view": 1})
		self.assertEqual(report["files"]["category_tree.csv"]["duplicate_rows"], 1)
		self.assertEqual(report["files"]["category_tree.csv"]["rejected_rows"], 1)
		self.assertEqual(report["files"]["item_properties_part1.csv"]["duplicate_rows"], 1)
		self.assertEqual(report["files"]["item_properties_part2.csv"]["duplicate_rows"], 3)
		self.assertEqual(report["processing_errors"]["transaction_missing_or_invalid_id"], 1)
		self.assertEqual(report["processing_errors"]["malformed_column_count"], 1)
		self.assertEqual(report["files"]["item_properties_part1.csv"]["missing_values"]["value"], 1)
		self.assertEqual(len(self._read_csv("item_properties.csv")), 2)
		stored_report = json.loads((self.processed_dir / "validation_report.json").read_text())
		self.assertIn("not DealoRa database IDs", stored_report["id_namespace"])

	def test_rejects_missing_required_columns_without_outputs(self):
		self._write(
			"events.csv",
			"timestamp,visitorid,event,itemid\n1433221332117,10,view,100\n",
		)
		with self.assertRaisesRegex(ValueError, "transactionid"):
			preprocess_retailrocket(self.raw_dir, self.processed_dir)
		self.assertEqual(list(self.processed_dir.iterdir()), [])

	def test_refuses_existing_output_without_modifying_it(self):
		existing = self.processed_dir / OUTPUT_FILES[0]
		existing.write_text("keep this file", encoding="utf-8")
		with self.assertRaises(FileExistsError):
			preprocess_retailrocket(self.raw_dir, self.processed_dir)
		self.assertEqual(existing.read_text(encoding="utf-8"), "keep this file")
		self.assertEqual(list(self.processed_dir.iterdir()), [existing])

	def test_rejects_missing_property_schema(self):
		self._write("item_properties_part2.csv", "timestamp,itemid,property\n1,2,brand\n")
		with self.assertRaisesRegex(ValueError, "value"):
			preprocess_retailrocket(self.raw_dir, self.processed_dir)
		self.assertEqual(list(self.processed_dir.iterdir()), [])

	def test_accepts_zero_values_as_valid_non_negative_ids(self):
		self.assertEqual(_integer(0), 0)
		self.assertEqual(_integer("0"), 0)
		self.assertEqual(_timestamp(0)[0], 0)


class RetailrocketEvaluationTests(unittest.TestCase):
	def test_training_refuses_to_overwrite_existing_artifacts(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			interactions = root / "interactions.csv"
			interactions.write_text(
				"visitor_id,item_id,event_type,timestamp_ms\n1,10,view,1\n2,11,purchase,2\n",
				encoding="utf-8",
			)
			existing = root / MODEL_FILE
			existing.write_text("keep model", encoding="utf-8")
			with self.assertRaises(FileExistsError):
				train_evaluate(interactions, root, k=2)
			self.assertEqual(existing.read_text(encoding="utf-8"), "keep model")
			self.assertFalse((root / RESULTS_FILE).exists())

	def test_ranking_metrics_use_documented_fixed_k_denominator(self):
		metrics = ranking_metrics([10, 20, 30], {20, 30}, 3)
		self.assertAlmostEqual(metrics["precision"], 2 / 3)
		self.assertEqual(metrics["recall"], 1.0)
		self.assertGreater(metrics["ndcg"], 0.0)
		self.assertLess(metrics["ndcg"], 1.0)

	def test_training_accepts_case_insensitive_event_types(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			interactions = root / "interactions.csv"
			interactions.write_text(
				"visitor_id,item_id,event_type,timestamp_ms\n"
				"0,0,VIEW,1\n"
				"0,1,CART,2\n"
				"1,0,PURCHASE,3\n",
				encoding="utf-8",
			)
			results = train_evaluate(interactions, root, k=2)
			self.assertEqual(results["split"]["train_interactions"], 2)
			self.assertEqual(results["split"]["test_interactions"], 1)

	def test_training_uses_chronological_holdout_and_saves_measured_results(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			interactions = root / "interactions.csv"
			interactions.write_text(
				"visitor_id,item_id,event_type,timestamp_ms,timestamp_utc,transaction_id\n"
				"1,10,view,1,,\n"
				"1,11,view,2,,\n"
				"1,11,cart,3,,\n"
				"1,10,purchase,4,,100\n"
				"2,10,view,5,,\n"
				"2,12,view,6,,\n"
				"2,12,cart,7,,\n"
				"2,11,view,8,,\n"
				"1,12,purchase,9,,101\n"
				"3,13,purchase,9,,102\n",
				encoding="utf-8",
			)
			results = train_evaluate(interactions, root, k=2)
			self.assertEqual(results["split"]["train_interactions"], 8)
			self.assertEqual(results["split"]["test_interactions"], 2)
			self.assertEqual(results["split"]["cutoff_timestamp_ms"], 9)
			self.assertEqual(results["split"]["evaluation_users"], 2)
			self.assertEqual(results["split"]["cold_start_evaluation_users"], 1)
			self.assertEqual(results["metrics"]["evaluation_users"], 2)
			self.assertIsNotNone(results["metrics"]["precision_at_k"])
			self.assertIsNotNone(results["metrics"]["recall_at_k"])
			self.assertIsNotNone(results["metrics"]["ndcg_at_k"])
			self.assertTrue((root / MODEL_FILE).is_file())
			self.assertTrue((root / RESULTS_FILE).is_file())
			self.assertIn("never mapped to DealoRa IDs", results["id_namespace"])


if __name__ == "__main__":
	unittest.main()