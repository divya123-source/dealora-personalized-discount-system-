"""Ranking and sampled-candidate classification metrics for recommendation evaluation."""

import math


def ranking_metrics(recommended, relevant, k):
	"""Return Precision@K, Recall@K, NDCG@K, MAP@K, and HitRate@K for one user."""
	if k < 1:
		raise ValueError("k must be positive")
	relevant = set(relevant)
	selected = list(recommended[:k])
	hits = sum(item_id in relevant for item_id in selected)
	precision = hits / float(k)
	recall = hits / float(len(relevant)) if relevant else 0.0
	hit_rate = 1.0 if hits else 0.0
	dcg = sum(
		1.0 / math.log2(rank + 2)
		for rank, item_id in enumerate(selected)
		if item_id in relevant
	)
	ideal_hits = min(k, len(relevant))
	idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
	ndcg = dcg / idcg if idcg else 0.0
	precision_sum = 0.0
	seen_hits = 0
	for rank, item_id in enumerate(selected, start=1):
		if item_id in relevant:
			seen_hits += 1
			precision_sum += seen_hits / float(rank)
	denominator = min(k, len(relevant)) if relevant else 0
	map_score = precision_sum / float(denominator) if denominator else 0.0
	return {
		"precision": precision,
		"recall": recall,
		"ndcg": ndcg,
		"map": map_score,
		"hit_rate": hit_rate,
		"hits": hits,
	}


def mean_metrics(rows):
	if not rows:
		return {
			"evaluation_users": 0,
			"precision": None,
			"recall": None,
			"ndcg": None,
			"map": None,
			"hit_rate": None,
		}
	keys = ("precision", "recall", "ndcg", "map", "hit_rate")
	totals = {key: 0.0 for key in keys}
	for row in rows:
		for key in keys:
			totals[key] += float(row[key])
	count = float(len(rows))
	return {
		"evaluation_users": len(rows),
		**{key: totals[key] / count for key in keys},
	}


def metric_mean_std(fold_values):
	values = [value for value in fold_values if value is not None]
	if not values:
		return {"mean": None, "std": None, "n": 0}
	mean = sum(values) / float(len(values))
	if len(values) == 1:
		return {"mean": mean, "std": 0.0, "n": 1}
	variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
	return {"mean": mean, "std": math.sqrt(variance), "n": len(values)}


def confusion_counts(labels, predictions):
	true_positive = false_positive = true_negative = false_negative = 0
	for label, prediction in zip(labels, predictions):
		if label and prediction:
			true_positive += 1
		elif (not label) and prediction:
			false_positive += 1
		elif (not label) and (not prediction):
			true_negative += 1
		else:
			false_negative += 1
	return {
		"true_positive": true_positive,
		"false_positive": false_positive,
		"true_negative": true_negative,
		"false_negative": false_negative,
	}


def classification_metrics(labels, predictions):
	"""Binary classification metrics. Undefined scores are returned as None, never invented."""
	counts = confusion_counts(labels, predictions)
	total = len(labels)
	accuracy = (
		(counts["true_positive"] + counts["true_negative"]) / float(total)
		if total else None
	)
	predicted_positive = counts["true_positive"] + counts["false_positive"]
	actual_positive = counts["true_positive"] + counts["false_negative"]
	precision = (
		counts["true_positive"] / float(predicted_positive)
		if predicted_positive else None
	)
	recall = counts["true_positive"] / float(actual_positive) if actual_positive else None
	if precision is None or recall is None or (precision + recall) == 0:
		f1 = None
	else:
		f1 = 2.0 * precision * recall / (precision + recall)
	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"support": total,
		**counts,
		"setting": (
			"Sampled-candidate binary labels (relevant test purchases vs. "
			"non-purchased candidates). Not catalog-wide accuracy."
		),
	}


def roc_auc_score(labels, scores):
	"""Trapezoidal ROC-AUC from ranked scores. Returns None if a class is missing."""
	positives = [score for label, score in zip(labels, scores) if label]
	negatives = [score for label, score in zip(labels, scores) if not label]
	if not positives or not negatives:
		return None
	greater = 0.0
	tied = 0.0
	for positive in positives:
		for negative in negatives:
			if positive > negative:
				greater += 1.0
			elif positive == negative:
				tied += 1.0
	return (greater + 0.5 * tied) / float(len(positives) * len(negatives))


def threshold_predictions(scores, threshold):
	return [score >= threshold for score in scores]
